import tensorflow as tf

class RewardPredictor:
    def __init__( self, ActionShape, StateShape ):
        self.sess = tf.InteractiveSession()
        self.Layers = [100, 50, 1]
        self.Ashape = ActionShape
        self.Sshape = StateShape
        self.size = self.Ashape[0] + self.Sshape[0]
        
        self.t = tf.placeholder( tf.float32 )
        self.L = []
        self.a = tf.placeholder( tf.float32, shape = ( self.Ashape ) )
        self.s = tf.placeholder( tf.float32, shape = ( self.Sshape ) )
        self._x = tf.concat( [self.a, self.s], 0)
        self.x = tf.reshape( self._x, ( 1, self.size ) )

        self.L = []
        self.L.append(
            self.dense( self.x, ( self.size, self.Layers[0] ) )
            )
        self.L.append( tf.nn.relu(self.L[-1]) )
        
        self.L.append(
            self.dense( self.L[-1], ( self.Layers[0], self.Layers[1] ) )
            )
        self.L.append( tf.nn.relu(self.L[-1]) )

        self.L.append(
            self.dense( self.L[-1], ( self.Layers[1], self.Layers[2] ) )
            )
        self.L.append( tf.nn.sigmoid( self.L[-1] ) )        
        
        self.loss = tf.losses.mean_squared_error( self.t, self.L[-1] )

        self.train = tf.train.AdamOptimizer( 1e-4 ).minimize( self.loss )
        
        self.sess.run(tf.global_variables_initializer())

    def getReward( self, action, state ):
        feed_dict = { self.a: action, self.s: state }
        return self.sess.run( self.L[-1], feed_dict )

    def trainStep( self, action, state, target):
        feed_dict = { self.a: action, self.s: state, self.t: target}
        self.train.run( feed_dict = feed_dict )
        
    def weight(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def dense( self, x, shape ):
        W = self.weight( shape )
        B = self.bias( ( 1, shape[1] ) ) 
        return tf.matmul( x, W ) + B
