from PIL import Image
import numpy as np

class pixel:
    def __init__( self, x, y, color ):
        self.x = x
        self.y = y
        self.z = 0
        self.color = color
    def move( self, x, y, z ):
        self.x += x
        self.y += y
        self.z += z

def getPixels( filepath, color = ( 255, 255, 255 ) ):
    im = Image.open( filepath )
    px = im.load( )
    pixels = []
    for y in range(im.height):
        for x in range(im.width):
            if px[x, y] != color:
                pixels.append( pixel( x, y, px[x, y] ) )
    return pixels

def clean( pix ):
    pixels = pix.copy( )
    for i in pixels:
        for j in pixels:
            if i != j and i.x == j.x and i.y == j.y:
                if j.z > i.z:
                    del i
                else:
                    del j
    return pixels

def getState( pixels ):
    state = []
    for p in clean( pixels ):
        state.append( [p.x, p.y, p.z] )

    arr = np.array( state )
    arr = arr.flatten()
    return arr

def takeAction( action, pixels ):
    for i in range(len(pixels)):
        pixels[i].move( action[3*i], action[3*i + 1], action[3*i + 2] )

def saveImg( filename, pixels, size, color = ( 255, 255, 255 ) ):
    im = Image.new( "RGB", size, color )
    px = im.load( )
    for p in clean(pixels):
        if p.x < size[0] and p.x >= 0 and p.y >= 0 and p.y < size[1]:
            px[ p.x, p.y ] = p.color
    im.save( filename )
