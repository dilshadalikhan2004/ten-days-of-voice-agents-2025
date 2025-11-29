import json
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Optional, Annotated

from dotenv import load_dotenv
from pydantic import Field
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
)

from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# Logging

logger = logging.getLogger("dnd_game_master")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

load_dotenv(".env.local")


# Game State

@dataclass
class PlayerCharacter:
    name: str = "Adventurer"
    hp: int = 100
    max_hp: int = 100
    strength: int = 10
    intelligence: int = 10
    luck: int = 10
    inventory: list = field(default_factory=list)
    location: str = "Village Square"
    status: str = "Healthy"

@dataclass
class GameState:
    player: PlayerCharacter = field(default_factory=PlayerCharacter)
    story_progress: list = field(default_factory=list)
    current_scene: str = "start"
    game_started: bool = False
    selected_scenario: str = ""
    npcs: dict = field(default_factory=dict)  # {name: {status, attitude, location}}
    active_quests: list = field(default_factory=list)
    completed_quests: list = field(default_factory=list)

@dataclass
class Userdata:
    game_state: GameState = field(default_factory=GameState)


# Game Tools

@function_tool
async def roll_dice(
    ctx: RunContext[Userdata],
    sides: Annotated[int, Field(description="Number of sides on the dice (default 20)", default=20)] = 20,
) -> str:
    """Roll a dice for skill checks and random events."""
    result = random.randint(1, sides)
    logger.info(f"🎲 Dice roll: {result} (d{sides})")
    return f"You rolled a {result} on a d{sides}!"

@function_tool
async def check_inventory(
    ctx: RunContext[Userdata],
) -> str:
    """Check what items the player is carrying."""
    inventory = ctx.userdata.game_state.player.inventory
    if not inventory:
        return "Your inventory is empty."
    return f"You are carrying: {', '.join(inventory)}"

@function_tool
async def check_status(
    ctx: RunContext[Userdata],
) -> str:
    """Check player's current health and status."""
    player = ctx.userdata.game_state.player
    game_state = ctx.userdata.game_state
    
    status = f"Name: {player.name}\nHP: {player.hp}/{player.max_hp}\nSTR: {player.strength} | INT: {player.intelligence} | LUCK: {player.luck}\nStatus: {player.status}\nLocation: {player.location}"
    
    if game_state.active_quests:
        status += f"\nActive Quests: {len(game_state.active_quests)}"
    if game_state.completed_quests:
        status += f"\nCompleted Quests: {len(game_state.completed_quests)}"
    
    return status

@function_tool
async def add_item(
    ctx: RunContext[Userdata],
    item: Annotated[str, Field(description="Item to add to inventory")],
) -> str:
    """Add an item to the player's inventory."""
    ctx.userdata.game_state.player.inventory.append(item)
    logger.info(f"📦 Added item: {item}")
    return f"You picked up: {item}"

@function_tool
async def update_hp(
    ctx: RunContext[Userdata],
    change: Annotated[int, Field(description="HP change (positive for healing, negative for damage)")],
) -> str:
    """Update player's health points."""
    player = ctx.userdata.game_state.player
    old_hp = player.hp
    player.hp = max(0, min(100, player.hp + change))
    
    if change > 0:
        logger.info(f"💚 HP healed: {old_hp} -> {player.hp}")
        return f"You gained {change} HP! Current HP: {player.hp}/100"
    else:
        logger.info(f"💔 HP damaged: {old_hp} -> {player.hp}")
        return f"You took {abs(change)} damage! Current HP: {player.hp}/100"

@function_tool
async def update_location(
    ctx: RunContext[Userdata],
    location: Annotated[str, Field(description="New location name")],
) -> str:
    """Update player's current location."""
    old_location = ctx.userdata.game_state.player.location
    ctx.userdata.game_state.player.location = location
    logger.info(f"🗺️ Location changed: {old_location} -> {location}")
    return f"You have moved to: {location}"

@function_tool
async def save_progress(
    ctx: RunContext[Userdata],
    event: Annotated[str, Field(description="Important story event to remember")],
) -> str:
    """Save important story progress."""
    ctx.userdata.game_state.story_progress.append(event)
    logger.info(f"📝 Story progress saved: {event}")
    return f"Progress saved: {event}"

@function_tool
async def restart_game(
    ctx: RunContext[Userdata],
) -> str:
    """Restart the adventure with a fresh character."""
    ctx.userdata.game_state = GameState()
    logger.info("🔄 Game restarted")
    return "Game restarted! Ready for a new adventure."

@function_tool
async def select_scenario(
    ctx: RunContext[Userdata],
    scenario: Annotated[str, Field(description="Scenario choice: fantasy, cyberpunk, or space")],
) -> str:
    """Select the adventure scenario."""
    scenarios = {
        "fantasy": "Middle-earth fantasy adventure",
        "cyberpunk": "Cyberpunk 2077-style city adventure", 
        "space": "Star Wars-style space opera"
    }
    
    scenario = scenario.lower()
    if scenario in scenarios:
        ctx.userdata.game_state.selected_scenario = scenario
        ctx.userdata.game_state.game_started = True
        logger.info(f"🎭 Scenario selected: {scenario}")
        return f"Scenario selected: {scenarios[scenario]}. Let the adventure begin!"
    else:
        return "Invalid scenario. Choose: fantasy, cyberpunk, or space."

@function_tool
async def skill_check(
    ctx: RunContext[Userdata],
    skill: Annotated[str, Field(description="Skill type: strength, intelligence, or luck")],
    difficulty: Annotated[int, Field(description="Difficulty modifier (0-10)", default=0)] = 0,
) -> str:
    """Perform a skill check with character attributes."""
    player = ctx.userdata.game_state.player
    base_roll = random.randint(1, 20)
    
    # Get attribute modifier
    attr_bonus = getattr(player, skill.lower(), 10) - 10
    total = base_roll + attr_bonus - difficulty
    
    if total >= 16:
        result = "Critical Success!"
    elif total >= 11:
        result = "Success"
    elif total >= 6:
        result = "Partial Success"
    else:
        result = "Failure"
    
    logger.info(f"🎲 Skill check ({skill}): {base_roll} + {attr_bonus} - {difficulty} = {total} ({result})")
    return f"Rolling {skill} check: {base_roll} + {attr_bonus} - {difficulty} = {total}. {result}!"

@function_tool
async def update_npc(
    ctx: RunContext[Userdata],
    name: Annotated[str, Field(description="NPC name")],
    status: Annotated[str, Field(description="NPC status (alive/dead/missing)")],
    attitude: Annotated[str, Field(description="NPC attitude (friendly/neutral/hostile)")],
) -> str:
    """Update or add an NPC to the world state."""
    ctx.userdata.game_state.npcs[name] = {
        "status": status,
        "attitude": attitude,
        "location": ctx.userdata.game_state.player.location
    }
    logger.info(f"👤 NPC updated: {name} ({status}, {attitude})")
    return f"NPC {name} is now {status} and {attitude}."

@function_tool
async def add_quest(
    ctx: RunContext[Userdata],
    quest: Annotated[str, Field(description="Quest description")],
) -> str:
    """Add a new quest to the active quests."""
    ctx.userdata.game_state.active_quests.append(quest)
    logger.info(f"📋 Quest added: {quest}")
    return f"New quest: {quest}"

@function_tool
async def complete_quest(
    ctx: RunContext[Userdata],
    quest: Annotated[str, Field(description="Quest to complete")],
) -> str:
    """Mark a quest as completed."""
    if quest in ctx.userdata.game_state.active_quests:
        ctx.userdata.game_state.active_quests.remove(quest)
        ctx.userdata.game_state.completed_quests.append(quest)
        logger.info(f"✅ Quest completed: {quest}")
        return f"Quest completed: {quest}"
    return f"Quest '{quest}' not found in active quests."

@function_tool
async def save_game(
    ctx: RunContext[Userdata],
) -> str:
    """Save the current game state to a JSON file."""
    import json
    from datetime import datetime
    
    game_data = {
        "timestamp": datetime.now().isoformat(),
        "player": {
            "name": ctx.userdata.game_state.player.name,
            "hp": ctx.userdata.game_state.player.hp,
            "max_hp": ctx.userdata.game_state.player.max_hp,
            "strength": ctx.userdata.game_state.player.strength,
            "intelligence": ctx.userdata.game_state.player.intelligence,
            "luck": ctx.userdata.game_state.player.luck,
            "inventory": ctx.userdata.game_state.player.inventory,
            "location": ctx.userdata.game_state.player.location,
            "status": ctx.userdata.game_state.player.status
        },
        "scenario": ctx.userdata.game_state.selected_scenario,
        "story_progress": ctx.userdata.game_state.story_progress,
        "npcs": ctx.userdata.game_state.npcs,
        "active_quests": ctx.userdata.game_state.active_quests,
        "completed_quests": ctx.userdata.game_state.completed_quests
    }
    
    filename = f"game_save_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(filename, 'w') as f:
            json.dump(game_data, f, indent=2)
        logger.info(f"💾 Game saved: {filename}")
        return f"Game saved as {filename}"
    except Exception as e:
        return f"Failed to save game: {e}"

@function_tool
async def check_session_status(
    ctx: RunContext[Userdata],
) -> str:
    """Check if this is a new session or continuing session."""
    game_state = ctx.userdata.game_state
    
    # Check if there's existing progress
    has_progress = (
        game_state.story_progress or 
        game_state.selected_scenario or 
        game_state.player.inventory or 
        game_state.player.hp != 100 or
        game_state.active_quests or
        game_state.completed_quests
    )
    
    if has_progress:
        summary = f"Welcome back, {game_state.player.name}! "
        summary += f"You're at {game_state.player.location} with {game_state.player.hp}/{game_state.player.max_hp} HP. "
        if game_state.selected_scenario:
            summary += f"Continuing your {game_state.selected_scenario} adventure. "
        if game_state.active_quests:
            summary += f"You have {len(game_state.active_quests)} active quest(s). "
        if game_state.story_progress:
            summary += f"Last event: {game_state.story_progress[-1]}. "
        logger.info("🔄 Resuming existing session")
        return summary + "Ready to continue your adventure!"
    else:
        logger.info("✨ New session started")
        return "Greetings, brave adventurer! Welcome to the realm of endless possibilities. I am your Game Master, ready to guide you through epic tales of heroism and adventure."

@function_tool
async def load_game(
    ctx: RunContext[Userdata],
    filename: Annotated[str, Field(description="Save file name to load")],
) -> str:
    """Load a previously saved game state."""
    import json
    
    try:
        with open(filename, 'r') as f:
            game_data = json.load(f)
        
        # Restore player data
        player_data = game_data["player"]
        ctx.userdata.game_state.player.name = player_data["name"]
        ctx.userdata.game_state.player.hp = player_data["hp"]
        ctx.userdata.game_state.player.max_hp = player_data["max_hp"]
        ctx.userdata.game_state.player.strength = player_data["strength"]
        ctx.userdata.game_state.player.intelligence = player_data["intelligence"]
        ctx.userdata.game_state.player.luck = player_data["luck"]
        ctx.userdata.game_state.player.inventory = player_data["inventory"]
        ctx.userdata.game_state.player.location = player_data["location"]
        ctx.userdata.game_state.player.status = player_data["status"]
        
        # Restore game state
        ctx.userdata.game_state.selected_scenario = game_data["scenario"]
        ctx.userdata.game_state.story_progress = game_data["story_progress"]
        ctx.userdata.game_state.npcs = game_data["npcs"]
        ctx.userdata.game_state.active_quests = game_data["active_quests"]
        ctx.userdata.game_state.completed_quests = game_data["completed_quests"]
        ctx.userdata.game_state.game_started = True
        
        logger.info(f"💾 Game loaded: {filename}")
        return f"Game loaded successfully! Welcome back, {ctx.userdata.game_state.player.name}. You're at {ctx.userdata.game_state.player.location}."
    except Exception as e:
        return f"Failed to load game: {e}"

@function_tool
async def end_game(
    ctx: RunContext[Userdata],
) -> str:
    """End the current adventure and provide a summary."""
    progress = ctx.userdata.game_state.story_progress
    player = ctx.userdata.game_state.player
    
    summary = f"Adventure Complete! Your hero {player.name} ended with {player.hp}/{player.max_hp} HP at {player.location}."
    if ctx.userdata.game_state.completed_quests:
        summary += f" Completed quests: {len(ctx.userdata.game_state.completed_quests)}"
    if progress:
        summary += f" Key events: {', '.join(progress[-3:])}"
    
    logger.info("🏁 Game ended")
    return summary + " Thanks for playing! Say 'restart' for a new adventure."

# Agent Definition

class GameMasterAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
            You are a D&D-style Game Master who can run adventures in multiple universes.
            
            PERSONA & TONE:
            - You are an experienced, dramatic storyteller
            - Use vivid descriptions and immersive language
            - Create tension and excitement
            - Be encouraging but present real challenges
            
            GAME RULES:
            1. FIRST MESSAGE: Always call check_session_status to greet properly (new vs returning player)
            2. ALWAYS end each response with 2-4 specific choices for the player
            3. Format choices as: "You can: A) [action], B) [action], C) [action], or tell me something else you'd like to do."
            4. Use the tools to track player state (HP, inventory, location)
            5. Call roll_dice for risky actions and skill checks
            6. Remember past events using save_progress
            7. Keep scenes engaging with 2-4 sentences of description
            
            CHOICE EXAMPLES:
            - "You can: A) Enter the tavern, B) Approach the merchant, C) Head to the forest path"
            - "You can: A) Attack with your sword, B) Try to sneak past, C) Attempt to negotiate"
            - "You can: A) Pick up the glowing orb, B) Search the room further, C) Leave immediately"
            
            SCENARIOS:
            1. FANTASY: Middle-earth adventure (Hobbiton -> Forest -> Cave -> Boss)
            2. CYBERPUNK: Neo-Tokyo 2077 (Streets -> Club -> Corporate Tower -> Hacker Boss)
            3. SPACE: Star Wars galaxy (Cantina -> Ship -> Space Station -> Sith Lord)
            
            SCENARIO SELECTION:
            - If no scenario selected, offer: "Choose your adventure: A) Fantasy, B) Cyberpunk, C) Space"
            - Use select_scenario tool when player chooses
            - Adapt all descriptions, NPCs, and items to match the selected scenario
            
            MECHANICS:
            - Use skill_check for attribute-based rolls (strength/intelligence/luck)
            - Use roll_dice for general random events
            - Track NPCs with update_npc (status: alive/dead, attitude: friendly/hostile)
            - Manage quests with add_quest and complete_quest
            - Character has STR/INT/LUCK stats (10 is average, affects skill checks)
            - HP starts at 100, damage 10-30, healing 20-50
            - Use save_game for important story moments
            
            Remember: Always give players clear options to choose from!
            """,
            tools=[roll_dice, skill_check, check_inventory, check_status, add_item, update_hp, update_location, save_progress, update_npc, add_quest, complete_quest, save_game, load_game, check_session_status, restart_game, end_game, select_scenario],
        )


# Entrypoint

def prewarm(proc: JobProcess):
    try:
        proc.userdata["vad"] = silero.VAD.load()
    except Exception:
        logger.warning("VAD prewarm failed; continuing without preloaded VAD.")

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info("\n" + "🧙" * 12)
    logger.info("🎲 STARTING D&D GAME MASTER - MIDDLE-EARTH ADVENTURE")

    userdata = Userdata()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-marcus",
            style="Conversational",
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata.get("vad"),
        userdata=userdata,
    )

    await session.start(
        agent=GameMasterAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
