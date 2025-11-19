"""
API routes for AI training management and visualization.
"""
import json
from typing import Dict, List
from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from app.ai.phased_trainer import PhasedTrainer


# Global trainer instance
trainer: PhasedTrainer = None
training_task_active = False


class TrainingConfig(BaseModel):
    """Configuration for starting a new training session."""
    episodes_per_phase: int = 10000
    checkpoint_interval: int = 1000
    games_to_record: int = 3
    num_simulations: int = 100
    board_size: int = 9
    num_channels: int = 128
    num_res_blocks: int = 5


class PhaseRequest(BaseModel):
    """Request to start a specific training phase."""
    phase_number: int
    num_episodes: int
    num_simulations: int = 100
    batch_size: int = 32


router = APIRouter(prefix="/api/training", tags=["training"])


@router.post("/initialize")
async def initialize_training(config: TrainingConfig):
    """
    Initialize training system and save initial weights.
    Records initial self-play games before training.
    """
    global trainer

    try:
        trainer = PhasedTrainer(
            board_size=config.board_size,
            num_channels=config.num_channels,
            num_res_blocks=config.num_res_blocks
        )

        trainer.initialize_training(
            episodes_per_phase=config.episodes_per_phase,
            checkpoint_interval=config.checkpoint_interval,
            games_to_record_per_checkpoint=config.games_to_record,
            num_simulations=config.num_simulations
        )

        return {
            "status": "initialized",
            "message": f"Training initialized. Initial checkpoint and {config.games_to_record} games recorded.",
            "config": config.dict()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize training: {str(e)}")


@router.post("/start-phase")
async def start_training_phase(phase_request: PhaseRequest, background_tasks: BackgroundTasks):
    """
    Start a specific training phase in the background.
    Training will run asynchronously and can be paused.
    """
    global trainer, training_task_active

    if trainer is None:
        raise HTTPException(status_code=400, detail="Training not initialized. Call /initialize first.")

    if training_task_active:
        raise HTTPException(status_code=400, detail="Training already in progress. Pause or wait for completion.")

    # Start training in background
    training_task_active = True

    def run_phase():
        global training_task_active
        try:
            trainer.train_phase(
                phase_number=phase_request.phase_number,
                num_episodes=phase_request.num_episodes,
                checkpoint_interval=trainer.phase_config['checkpoint_interval'],
                num_simulations=phase_request.num_simulations,
                batch_size=phase_request.batch_size
            )
        finally:
            training_task_active = False

    background_tasks.add_task(run_phase)

    return {
        "status": "started",
        "phase": phase_request.phase_number,
        "num_episodes": phase_request.num_episodes,
        "message": f"Phase {phase_request.phase_number} training started in background"
    }


@router.post("/pause")
async def pause_training():
    """Pause the current training phase."""
    global trainer

    if trainer is None:
        raise HTTPException(status_code=400, detail="Training not initialized")

    if not trainer.is_training:
        raise HTTPException(status_code=400, detail="Training is not currently active")

    trainer.pause_training()

    return {
        "status": "paused",
        "episode": trainer.current_episode,
        "phase": trainer.current_phase
    }


@router.post("/resume")
async def resume_training():
    """Resume paused training."""
    global trainer

    if trainer is None:
        raise HTTPException(status_code=400, detail="Training not initialized")

    if trainer.is_training:
        raise HTTPException(status_code=400, detail="Training is already active")

    trainer.resume_training()

    return {
        "status": "resumed",
        "episode": trainer.current_episode,
        "phase": trainer.current_phase
    }


@router.get("/status")
async def get_training_status():
    """Get current training status and progress."""
    global trainer, training_task_active

    if trainer is None:
        return {
            "initialized": False,
            "message": "Training not initialized"
        }

    status = trainer.get_training_status()
    status['task_active'] = training_task_active
    status['initialized'] = True

    return status


@router.get("/checkpoints")
async def list_checkpoints():
    """List all saved checkpoints with metadata."""
    global trainer

    if trainer is None:
        raise HTTPException(status_code=400, detail="Training not initialized")

    checkpoints = []
    checkpoint_dir = Path(trainer.checkpoint_dir)

    for metadata_file in sorted(checkpoint_dir.glob("*_metadata.json")):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            checkpoints.append(metadata)

    return {
        "total_checkpoints": len(checkpoints),
        "checkpoints": checkpoints
    }


@router.get("/checkpoint/{checkpoint_name}")
async def get_checkpoint_details(checkpoint_name: str):
    """Get detailed information about a specific checkpoint."""
    global trainer

    if trainer is None:
        raise HTTPException(status_code=400, detail="Training not initialized")

    metadata_file = Path(trainer.checkpoint_dir) / f"{checkpoint_name}_metadata.json"

    if not metadata_file.exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint '{checkpoint_name}' not found")

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    return metadata


@router.get("/checkpoint/{checkpoint_name}/weights")
async def get_checkpoint_weights(checkpoint_name: str, layer: str = None):
    """
    Get weight statistics for a checkpoint.
    Optionally filter by layer name.
    """
    global trainer

    if trainer is None:
        raise HTTPException(status_code=400, detail="Training not initialized")

    metadata_file = Path(trainer.checkpoint_dir) / f"{checkpoint_name}_metadata.json"

    if not metadata_file.exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint '{checkpoint_name}' not found")

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    weight_stats = metadata.get('weight_stats', {})

    if layer:
        if layer not in weight_stats:
            raise HTTPException(status_code=404, detail=f"Layer '{layer}' not found in checkpoint")
        return {layer: weight_stats[layer]}

    return weight_stats


@router.get("/games")
async def list_recorded_games():
    """List all recorded game files."""
    global trainer

    if trainer is None:
        raise HTTPException(status_code=400, detail="Training not initialized")

    games_dir = Path(trainer.games_dir)
    game_files = []

    for game_file in sorted(games_dir.glob("*.json")):
        with open(game_file, 'r') as f:
            games = json.load(f)
            game_files.append({
                'filename': game_file.name,
                'num_games': len(games),
                'checkpoint_episode': games[0]['checkpoint_episode'] if games else None,
                'timestamp': games[0]['timestamp'] if games else None
            })

    return {
        "total_files": len(game_files),
        "game_files": game_files
    }


@router.get("/games/{filename}")
async def get_recorded_games(filename: str):
    """Get recorded games from a specific file."""
    global trainer

    if trainer is None:
        raise HTTPException(status_code=400, detail="Training not initialized")

    game_file = Path(trainer.games_dir) / filename

    if not game_file.exists():
        raise HTTPException(status_code=404, detail=f"Game file '{filename}' not found")

    with open(game_file, 'r') as f:
        games = json.load(f)

    return {
        "filename": filename,
        "num_games": len(games),
        "games": games
    }


@router.get("/metrics")
async def get_training_metrics(limit: int = 100):
    """Get training metrics (losses, etc.)."""
    global trainer

    if trainer is None:
        raise HTTPException(status_code=400, detail="Training not initialized")

    metrics = trainer.training_metrics[-limit:] if trainer.training_metrics else []

    return {
        "total_metrics": len(trainer.training_metrics),
        "returned_metrics": len(metrics),
        "metrics": metrics
    }


@router.get("/weight-comparison")
async def compare_weights(checkpoint1: str, checkpoint2: str):
    """
    Compare weights between two checkpoints.
    Shows differences in statistics.
    """
    global trainer

    if trainer is None:
        raise HTTPException(status_code=400, detail="Training not initialized")

    # Load metadata for both checkpoints
    metadata1_file = Path(trainer.checkpoint_dir) / f"{checkpoint1}_metadata.json"
    metadata2_file = Path(trainer.checkpoint_dir) / f"{checkpoint2}_metadata.json"

    if not metadata1_file.exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint '{checkpoint1}' not found")
    if not metadata2_file.exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint '{checkpoint2}' not found")

    with open(metadata1_file, 'r') as f:
        metadata1 = json.load(f)
    with open(metadata2_file, 'r') as f:
        metadata2 = json.load(f)

    weight_stats1 = metadata1.get('weight_stats', {})
    weight_stats2 = metadata2.get('weight_stats', {})

    # Calculate differences
    comparison = {}
    for layer in weight_stats1.keys():
        if layer in weight_stats2:
            comparison[layer] = {
                'mean_diff': weight_stats2[layer]['mean'] - weight_stats1[layer]['mean'],
                'std_diff': weight_stats2[layer]['std'] - weight_stats1[layer]['std'],
                'checkpoint1': weight_stats1[layer],
                'checkpoint2': weight_stats2[layer]
            }

    return {
        'checkpoint1': {
            'name': checkpoint1,
            'episode': metadata1['episode'],
            'timestamp': metadata1['timestamp']
        },
        'checkpoint2': {
            'name': checkpoint2,
            'episode': metadata2['episode'],
            'timestamp': metadata2['timestamp']
        },
        'weight_comparison': comparison
    }


@router.get("/architecture")
async def get_network_architecture():
    """Get neural network architecture information."""
    global trainer

    if trainer is None:
        raise HTTPException(status_code=400, detail="Training not initialized")

    architecture = {
        'board_size': trainer.board_size,
        'input_shape': [3, trainer.board_size, trainer.board_size],
        'layers': []
    }

    # Extract layer information
    for name, module in trainer.neural_net.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            layer_info = {
                'name': name,
                'type': module.__class__.__name__
            }

            # Add parameter info if available
            if hasattr(module, 'weight') and module.weight is not None:
                layer_info['weight_shape'] = list(module.weight.shape)
                layer_info['num_parameters'] = module.weight.numel()

            if hasattr(module, 'bias') and module.bias is not None:
                layer_info['bias_shape'] = list(module.bias.shape)

            architecture['layers'].append(layer_info)

    # Calculate total parameters
    total_params = sum(p.numel() for p in trainer.neural_net.parameters())
    trainable_params = sum(p.numel() for p in trainer.neural_net.parameters() if p.requires_grad)

    architecture['total_parameters'] = total_params
    architecture['trainable_parameters'] = trainable_params

    return architecture
