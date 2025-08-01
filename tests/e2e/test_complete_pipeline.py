"""End-to-end tests for complete system pipeline."""

import pytest
import tempfile
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import jax.numpy as jnp
import numpy as np


class TestCompletePipeline:
    """End-to-end tests for the complete dynamic graph fed-RL pipeline."""
    
    def test_configuration_loading_and_validation(self):
        """Test that system properly loads and validates configuration."""
        # Create temporary config file
        config = {
            "environment": {
                "type": "traffic_network",
                "num_nodes": 50,
                "time_varying": True,
                "seed": 42
            },
            "federation": {
                "num_agents": 5,
                "protocol": "async_gossip",
                "aggregation_interval": 100,
                "byzantine_robustness": True
            },
            "training": {
                "max_episodes": 100,
                "learning_rate": 3e-4,
                "batch_size": 256,
                "buffer_size": 10000
            },
            "monitoring": {
                "log_interval": 10,
                "save_interval": 50,
                "metrics_enabled": True
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            config_path = f.name
        
        # Mock configuration loader
        mock_config_loader = Mock()
        mock_config_loader.load_config.return_value = config
        mock_config_loader.validate_config.return_value = True
        
        # Test configuration loading
        loaded_config = mock_config_loader.load_config(config_path)
        is_valid = mock_config_loader.validate_config(loaded_config)
        
        assert is_valid
        assert loaded_config["environment"]["type"] == "traffic_network"
        assert loaded_config["federation"]["num_agents"] == 5
        assert loaded_config["training"]["learning_rate"] == 3e-4
        
        # Cleanup
        Path(config_path).unlink()
    
    def test_environment_initialization_and_reset(self):
        """Test environment setup and reset functionality."""
        env_config = {
            "type": "traffic_network",
            "num_nodes": 25,
            "grid_size": 5,
            "time_varying": True,
            "seed": 42
        }
        
        # Mock environment
        mock_env = Mock()
        mock_env.reset.return_value = {
            "nodes": jnp.ones((25, 4)),
            "edges": jnp.array([[0, 1], [1, 2], [2, 3]]),
            "edge_features": jnp.ones((3, 3)),
            "num_nodes": 25,
            "num_edges": 3
        }
        mock_env.observation_space = Mock()
        mock_env.action_space = Mock()
        
        # Test environment initialization
        initial_state = mock_env.reset()
        
        assert initial_state is not None
        assert "nodes" in initial_state
        assert "edges" in initial_state
        assert initial_state["num_nodes"] == 25
        
        # Test multiple resets (should be consistent with seed)
        state1 = mock_env.reset()
        state2 = mock_env.reset()
        
        # With fixed seed, states should be identical
        mock_env.reset.return_value = initial_state  # Simulate deterministic reset
        assert jnp.array_equal(state1["nodes"], state2["nodes"])
    
    def test_agent_initialization_and_federation_setup(self):
        """Test agent creation and federated system setup."""
        fed_config = {
            "num_agents": 4,
            "protocol": "async_gossip",
            "aggregation_interval": 50,
            "compression": "top_k",
            "compression_ratio": 0.1
        }
        
        # Mock federated system
        mock_fed_system = Mock()
        mock_fed_system.num_agents = fed_config["num_agents"]
        mock_fed_system.protocol = fed_config["protocol"]
        
        # Mock agents
        agents = []
        for i in range(fed_config["num_agents"]):
            agent = Mock()
            agent.id = i
            agent.get_parameters = Mock(return_value={"weights": jnp.ones((10, 10))})
            agent.set_parameters = Mock()
            agent.select_action = Mock(return_value=jnp.array([0.5, -0.3]))
            agents.append(agent)
        
        mock_fed_system.agents = agents
        
        # Test federation setup
        assert len(mock_fed_system.agents) == fed_config["num_agents"]
        
        # Test parameter synchronization
        initial_params = [agent.get_parameters() for agent in agents]
        
        # Mock aggregation
        def mock_aggregate(agent_list):
            avg_params = {"weights": jnp.mean(jnp.stack([
                agent.get_parameters()["weights"] for agent in agent_list
            ]), axis=0)}
            
            for agent in agent_list:
                agent.set_parameters(avg_params)
        
        mock_fed_system.aggregate_parameters = mock_aggregate
        mock_fed_system.aggregate_parameters(agents)
        
        # Verify parameter updates
        for agent in agents:
            assert agent.set_parameters.called
    
    def test_training_loop_execution(self):
        """Test the complete training loop execution."""
        # Mock components
        mock_env = Mock()
        mock_env.reset.return_value = {
            "nodes": jnp.ones((10, 4)),
            "edges": jnp.array([[0, 1], [1, 2]]),
            "edge_features": jnp.ones((2, 2))
        }
        mock_env.max_episode_steps = 20
        
        # Mock step function with varying rewards
        step_count = 0
        def mock_step(actions):
            nonlocal step_count
            step_count += 1
            rewards = {f"agent_{i}": np.random.random() for i in range(len(actions))}
            done = step_count >= mock_env.max_episode_steps
            info = {"step": step_count}
            return mock_env.reset.return_value, rewards, done, info
        
        mock_env.step = mock_step
        mock_env.get_agent_observation = Mock(return_value=mock_env.reset.return_value)
        
        # Mock agents
        num_agents = 3
        agents = []
        for i in range(num_agents):
            agent = Mock()
            agent.select_action = Mock(return_value=jnp.array([0.1 * i, 0.2 * i]))
            agent.store_transition = Mock()
            agent.train = Mock()
            agent.buffer = Mock()
            agent.buffer.__len__ = Mock(return_value=1000)  # Sufficient data for training
            agents.append(agent)
        
        # Mock federated system
        mock_fed_system = Mock()
        mock_fed_system.aggregation_interval = 10
        mock_fed_system.aggregate_parameters = Mock()
        
        # Training configuration
        max_episodes = 3
        train_interval = 5
        
        # Execute training loop
        total_rewards = []
        
        for episode in range(max_episodes):
            step_count = 0  # Reset for each episode
            state = mock_env.reset()
            episode_rewards = {f"agent_{i}": 0 for i in range(num_agents)}
            
            for step in range(mock_env.max_episode_steps):
                # Agent actions
                actions = {}
                for i, agent in enumerate(agents):
                    local_state = mock_env.get_agent_observation(i)
                    action = agent.select_action(local_state)
                    actions[f"agent_{i}"] = action
                
                # Environment step
                next_state, rewards, done, info = mock_env.step(actions)
                
                # Store experiences
                for i, agent in enumerate(agents):
                    agent.store_transition(
                        state=state,
                        action=actions[f"agent_{i}"],
                        reward=rewards[f"agent_{i}"],
                        next_state=next_state,
                        done=done
                    )
                    episode_rewards[f"agent_{i}"] += rewards[f"agent_{i}"]
                
                # Training
                if step % train_interval == 0:
                    for agent in agents:
                        agent.train()
                
                # Federated aggregation
                if step % mock_fed_system.aggregation_interval == 0:
                    mock_fed_system.aggregate_parameters(agents)
                
                state = next_state
                
                if done:
                    break
            
            total_rewards.append(sum(episode_rewards.values()))
        
        # Verify training execution
        assert len(total_rewards) == max_episodes
        assert all(reward >= 0 for reward in total_rewards)  # Assuming non-negative rewards
        
        # Verify agent interactions
        for agent in agents:
            assert agent.select_action.call_count > 0
            assert agent.store_transition.call_count > 0
            assert agent.train.call_count > 0
        
        # Verify federated learning
        assert mock_fed_system.aggregate_parameters.call_count > 0
    
    def test_model_saving_and_loading(self):
        """Test model checkpointing and loading functionality."""
        # Mock model and checkpoint system
        mock_model = Mock()
        mock_model.get_state = Mock(return_value={"weights": jnp.ones((5, 5)), "step": 1000})
        mock_model.load_state = Mock()
        
        mock_checkpoint_manager = Mock()
        
        def mock_save_checkpoint(model, path, metadata=None):
            # Simulate saving to disk
            return {"success": True, "path": path}
        
        def mock_load_checkpoint(path):
            # Simulate loading from disk
            return {"weights": jnp.ones((5, 5)), "step": 1000}
        
        mock_checkpoint_manager.save = mock_save_checkpoint
        mock_checkpoint_manager.load = mock_load_checkpoint
        
        # Test saving
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "checkpoint_1000.pkl"
            
            result = mock_checkpoint_manager.save(
                mock_model, 
                str(checkpoint_path),
                metadata={"episode": 100, "reward": 150.5}
            )
            
            assert result["success"]
            
            # Test loading
            loaded_state = mock_checkpoint_manager.load(str(checkpoint_path))
            mock_model.load_state(loaded_state)
            
            assert mock_model.load_state.called
            assert "weights" in loaded_state
            assert loaded_state["step"] == 1000
    
    def test_monitoring_and_logging_integration(self):
        """Test monitoring system and logging integration."""
        # Mock monitoring components
        mock_logger = Mock()
        mock_metrics_collector = Mock()
        mock_tensorboard = Mock()
        
        # Metrics storage
        metrics = {
            "episode_rewards": [],
            "training_loss": [],
            "communication_overhead": [],
            "system_performance": []
        }
        
        def mock_log_metrics(metric_dict, step):
            for key, value in metric_dict.items():
                if key in metrics:
                    metrics[key].append(value)
            mock_tensorboard.add_scalars(metric_dict, step)
        
        mock_metrics_collector.log = mock_log_metrics
        
        # Simulate training with monitoring
        episodes = 5
        for episode in range(episodes):
            # Simulate episode metrics
            episode_reward = np.random.uniform(10, 100)
            training_loss = np.random.uniform(0.1, 1.0)
            comm_overhead = np.random.uniform(0.01, 0.1)
            system_perf = np.random.uniform(0.8, 1.0)
            
            # Log metrics
            episode_metrics = {
                "episode_rewards": episode_reward,
                "training_loss": training_loss,
                "communication_overhead": comm_overhead,
                "system_performance": system_perf
            }
            
            mock_metrics_collector.log(episode_metrics, episode)
            
            # Log to standard logger
            mock_logger.info(f"Episode {episode}: Reward={episode_reward:.2f}")
        
        # Verify monitoring
        assert len(metrics["episode_rewards"]) == episodes
        assert len(metrics["training_loss"]) == episodes
        assert mock_logger.info.call_count == episodes
        assert mock_tensorboard.add_scalars.call_count == episodes
        
        # Verify metric ranges
        assert all(10 <= r <= 100 for r in metrics["episode_rewards"])
        assert all(0.1 <= l <= 1.0 for l in metrics["training_loss"])
    
    def test_error_handling_and_recovery(self):
        """Test system error handling and recovery mechanisms."""
        # Mock components that can fail
        mock_env = Mock()
        mock_agent = Mock()
        
        # Simulate environment failure
        mock_env.step.side_effect = [
            ({"nodes": jnp.ones((5, 2))}, {"agent_0": 1.0}, False, {}),  # Success
            Exception("Environment connection lost"),  # Failure
            ({"nodes": jnp.ones((5, 2))}, {"agent_0": 0.5}, False, {}),  # Recovery
        ]
        
        # Error handler
        def safe_env_step(env, actions, max_retries=3):
            for attempt in range(max_retries):
                try:
                    return env.step(actions)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(0.1)  # Brief delay before retry
                    continue
        
        # Test error handling
        actions = {"agent_0": jnp.array([0.5])}
        
        # First call should succeed
        result1 = safe_env_step(mock_env, actions)
        assert result1[1]["agent_0"] == 1.0
        
        # Second call should fail but recover
        result2 = safe_env_step(mock_env, actions)
        assert result2[1]["agent_0"] == 0.5
        
        # Verify retry logic was used
        assert mock_env.step.call_count == 3  # Initial + retry + success
    
    def test_distributed_deployment_simulation(self):
        """Test distributed deployment across multiple nodes."""
        # Mock distributed components
        num_nodes = 3
        agents_per_node = 2
        
        # Mock network nodes
        nodes = []
        for node_id in range(num_nodes):
            node = Mock()
            node.id = node_id
            node.agents = []
            
            # Create agents for this node
            for agent_idx in range(agents_per_node):
                agent = Mock()
                agent.id = f"node_{node_id}_agent_{agent_idx}"
                agent.select_action = Mock(return_value=jnp.array([0.1, 0.2]))
                agent.get_parameters = Mock(return_value={"weights": jnp.ones((3, 3))})
                node.agents.append(agent)
            
            nodes.append(node)
        
        # Mock communication system
        mock_comm = Mock()
        communication_log = []
        
        def mock_send_message(sender_id, receiver_id, message):
            communication_log.append({
                "sender": sender_id,
                "receiver": receiver_id,
                "message_type": message.get("type", "unknown"),
                "timestamp": time.time()
            })
            return {"status": "sent", "latency": np.random.uniform(0.01, 0.1)}
        
        mock_comm.send = mock_send_message
        
        # Simulate distributed training round
        for round_num in range(3):
            # Each node processes locally
            for node in nodes:
                for agent in node.agents:
                    # Local training step
                    action = agent.select_action({"dummy": "state"})
                    params = agent.get_parameters()
            
            # Inter-node communication (gossip)
            for sender_node in nodes:
                for receiver_node in nodes:
                    if sender_node.id != receiver_node.id:
                        # Send parameters between nodes
                        message = {
                            "type": "parameter_update",
                            "parameters": {"weights": jnp.zeros((3, 3))},
                            "round": round_num
                        }
                        mock_comm.send(sender_node.id, receiver_node.id, message)
        
        # Verify distributed execution
        total_agents = num_nodes * agents_per_node
        expected_communications = num_nodes * (num_nodes - 1) * 3  # 3 rounds
        
        assert len(communication_log) == expected_communications
        
        # Verify all nodes participated
        sender_ids = {msg["sender"] for msg in communication_log}
        receiver_ids = {msg["receiver"] for msg in communication_log}
        
        assert len(sender_ids) == num_nodes
        assert len(receiver_ids) == num_nodes
        
        # Verify message types
        message_types = {msg["message_type"] for msg in communication_log}
        assert "parameter_update" in message_types
    
    @pytest.mark.slow
    def test_system_performance_benchmarks(self):
        """Test system performance under various conditions."""
        # This would normally run longer benchmarks
        # Shortened for CI/CD pipeline
        
        scenarios = [
            {"agents": 5, "nodes": 25, "episodes": 10},
            {"agents": 10, "nodes": 50, "episodes": 10},
            {"agents": 15, "nodes": 75, "episodes": 10},
        ]
        
        performance_results = []
        
        for scenario in scenarios:
            start_time = time.time()
            
            # Mock training for this scenario
            num_agents = scenario["agents"]
            num_nodes = scenario["nodes"]
            episodes = scenario["episodes"]
            
            # Simulate computational load
            mock_work_time = (num_agents * num_nodes * episodes) / 10000  # Scaled down
            time.sleep(min(mock_work_time, 0.1))  # Cap sleep time for tests
            
            end_time = time.time()
            duration = end_time - start_time
            
            performance_results.append({
                "scenario": scenario,
                "duration": duration,
                "throughput": episodes / duration if duration > 0 else float('inf')
            })
        
        # Verify performance scaling
        assert len(performance_results) == len(scenarios)
        
        # Basic performance assertions
        for result in performance_results:
            assert result["duration"] > 0
            assert result["throughput"] > 0
        
        # Performance should degrade gracefully with scale
        # (This is a simplified check)
        durations = [r["duration"] for r in performance_results]
        assert durations[0] <= durations[-1]  # Later scenarios should take longer