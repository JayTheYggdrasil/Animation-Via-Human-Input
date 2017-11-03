from tensorforce import Configuration
from tensorforce.agents import PPOAgent


def getAgent( shapeIn, shapeOut ):
    
    config = Configuration(
        batch_size=1,
        step_optimizer=dict(
            type='adam',
            learning_rate=1e-4
        )
    )

    # Create a Proximal Policy Optimization agent
    agent = PPOAgent(
        dict( type='float', shape = shapeIn[0] ),
        dict( type='float',  shape = shapeOut[0] ),
        [
            dict(type='dense', size=64),
        ],
        config
    )
    
    return agent
