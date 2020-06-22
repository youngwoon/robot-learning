from .actor_critic import Actor, Critic


def get_actor_critic(encoder_type, algo):
    actor = critic = None
    if encoder_type == "mlp":
        from .mlp_actor_critic import MlpActor, MlpCritic, NoisyMlpActor

        if algo == "ddpg":  # add exploratory noise to actor
            actor = NoisyMlpActor
        elif algo in ["bc"]:
            return MlpActor, None
        else:
            actor = MlpActor
        return actor, MlpCritic

    elif encoder_type == "cnn":
        from .cnn_actor_critic import CnnActor, CnnCritic

        if algo in ["bc"]:
            return CnnActor, None
        else:
            actor = CnnActor
        return actor, CnnCritic

    else:
        raise ValueError("--encoder_type %s is not supported." % encoder_type)
