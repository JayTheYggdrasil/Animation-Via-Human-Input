from enviromentDefs import getPixels, getState, takeAction, saveImg
from Agent import getAgent
from REstimator import RewardPredictor
import numpy as np
from PIL import Image

image = 'Example.jpg'
IM = Image.open( image )

pixels = getPixels( image )
print( len(pixels) )
state0 = getState( pixels )
stateSpace = state0.shape
actionSpace = stateSpace
print( stateSpace )
Frames = 10
Steps = 100
SPUp = 10

StateAction1 = []
StateAction2 = []
Pair = [ StateAction1, StateAction2 ]

R = RewardPredictor( stateSpace, actionSpace )

Agent = getAgent( stateSpace, actionSpace )
for step in range(Steps):
    
    pixels = getPixels( image )
    state = getState( pixels )
    for i in range( Frames ):
        action = Agent.act( state )
        takeAction( action, pixels )
        reward = R.getReward( action, state )[0][0]
        Agent.observe( i == Frames-1, reward )
        state = getState( pixels )
    
    if step % SPUp:
        for SA in Pair:
            pixels = getPixels( image )
            saveImg( "C1/Frame1.jpg", pixels, IM.size )
            saveImg( "C2/Frame1.jpg", pixels, IM.size )
            state0 = getState( pixels )
            noise = np.random.uniform( low = -1, size = stateSpace)
            state = state0 + noise
            for i in range( Frames ):
                action = Agent.act( state )
                SA.append( [ state, action ] )
                takeAction( action, pixels )
                reward = R.getReward( action, state )[0][0]
                Agent.observe( i == Frames-1, reward )
                state = getState( pixels )
                if np.array_equal( SA, Pair[0] ):
                    saveImg( "C1/Frame" + str( i + 2 ) + ".jpg", pixels, IM.size )
                else:
                    saveImg( "C2/Frame" + str( i + 2 ) + ".jpg", pixels, IM.size )
        Choice = int(input( '1 or 2? : ' ))
        
        if( Choice == 1 ):
            for i in Pair[0]:
                R.trainStep( i[1], i[0], 1 )
            for i in Pair[1]:
                R.trainStep( i[1], i[0], -1 )
                
        if( Choice == 2 ):
            for i in Pair[0]:
                R.trainStep( i[1], i[0], -1 )
            for i in Pair[1]:
                R.trainStep( i[1], i[0], 1 )
            




