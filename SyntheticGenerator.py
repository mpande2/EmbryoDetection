from PIL import Image, ImageDraw, ImageOps
import random
from math import floor
import os

def log(*args):
    print(*args)

class Utility:
    @staticmethod
    def randomizeWithinVariance(val,maxVariance=0.2):
        new_val = round(val * (1 + random.uniform(-maxVariance,0)))
        return (new_val)

class SyntheticGenerator:
    INVALID_LOCATION = (-1,-1)
    def __init__(self,viable_images,non_viable_images,viable_percent=0.5):
        self.viables = viable_images
        self.non_viables = non_viable_images
        self.viablePercent = viable_percent
        self.image_height = 1000
        self.image_width = 1000
        self.separation_chance = 0.05
        self.egg_fill = 0.6
        self.padding = 1

    def _findNextEmptySlot(self,grid):
        if len(grid) == 0 or len(grid[0]) == 0: return SyntheticGenerator.INVALID_LOCATION

        rows = len(grid)
        cols = len(grid[0])

        for i in range(self.padding,rows-self.padding):
            for j in range(self.padding,cols-self.padding):
                if grid[i][j] == 0: return (i,j)
        return SyntheticGenerator.INVALID_LOCATION

    def _getNextEgg(self):
        r = random.random()
        if r < self.viablePercent:
            return random.choice(self.non_viables)
        else:
            return random.choice(self.viables)

    def _getDirectionCoords(self,current_coord,rows=0,cols=0,dir=0):
        padding = self.padding
        match dir:
            case 3:
                return (max(current_coord[0]-1,padding),current_coord[1])
            case 1:
                return (min(current_coord[0]+1,cols-1),current_coord[1])
            case 0:
                return (current_coord[0],max(current_coord[1]-1,padding))
            case 2:
                return (current_coord[0],min(current_coord[1]+1,rows-1))
            

    def _nextSpawnLocation(self,grid,curr_location,rows=0,cols=0):
        # Egg spawns at a random location occassionally   
        padding = self.padding
        cols = cols - (padding)
        rows = rows - (padding)     
        rand_num = random.random()
        if curr_location == SyntheticGenerator.INVALID_LOCATION or (rand_num < self.separation_chance):
            log('cluster splitting...')
            retry_count = 0
            while True:
                next_location = (random.randint(padding,rows-1),random.randint(padding,cols-1))
                retry_count+=1
                if grid[next_location[0]][next_location[1]] == 0: return next_location
                if retry_count > rows * cols: return SyntheticGenerator.INVALID_LOCATION

        # Egg spwans around the current egg
        potential_next_locations = []
        for i in range(4):
            next_location = self._getDirectionCoords(curr_location,rows=rows,cols=cols,dir=i)
            
            if grid[next_location[0]][next_location[1]] == 0:
                potential_next_locations.append((next_location[0],next_location[1]))
        
        if len(potential_next_locations) == 0:
            log('finding next empty slot')
            return self._findNextEmptySlot(grid)
        else:
            next_location = random.choice(potential_next_locations)
            log(f'got next_locations: {next_location}')
            return next_location

    def generateGrid(self,rows=10,cols=10,show_grid=False,generate_animation=False):
        row_height, col_width = self.image_width//rows, self.image_height//cols
        location_grid = [[0 for _ in range(cols)] for _ in range(rows)]

        im = Image.new('RGBA', (self.image_width,self.image_height))
        backgroundIm = Image.open(os.path.join('background','bg_29.png')).resize(im.size).convert(im.mode)
        draw = ImageDraw.Draw(backgroundIm)

        if show_grid:
            for row in range(0,self.image_height,row_height):
                draw.line([(0,row),(self.image_width,row)], fill='#00ffff')

            for col in range(0,self.image_width,col_width):
                draw.line([(col,0),(col,self.image_height)], fill='#00ffff')

        animation_frames = []
        next_spawn_location = SyntheticGenerator.INVALID_LOCATION
        for _ in range(floor(self.egg_fill*(rows-self.padding*2)*(cols-self.padding*2))):
            next_spawn_location = self._nextSpawnLocation(location_grid,next_spawn_location,rows=rows,cols=cols)
            if (next_spawn_location[0] < 0) or (next_spawn_location[1] <0): break
            next_egg_radius = Utility.randomizeWithinVariance(col_width)


            next_egg = self._getNextEgg().resize((next_egg_radius,next_egg_radius)).convert('RGBA')
            backgroundIm.paste(next_egg,((next_spawn_location[1])*col_width,(next_spawn_location[0])*row_height),next_egg)
            location_grid[next_spawn_location[0]][next_spawn_location[1]] = 1
            if generate_animation:
                animation_frames.append(backgroundIm.copy())

        if generate_animation:    
            animation_frames[0].save('animation.gif',save_all=True,append_images=animation_frames[1:],duration=2)
        return backgroundIm

# Dev test
viable_image_dir = 'fertilized'
non_viable_image_dir = 'unfertilized'

viable_images = [Image.open(os.path.join(viable_image_dir,img)) for img in os.listdir(viable_image_dir) if img.endswith('png')]
non_viable_images = [Image.open(os.path.join(non_viable_image_dir,img)) for img in os.listdir(non_viable_image_dir) if img.endswith('png')]

frog_generator = SyntheticGenerator(viable_images,non_viable_images,viable_percent=0.6)

im = frog_generator.generateGrid(10,10,show_grid=True,generate_animation=False)
im.show()
