from PIL import Image, ImageDraw, ImageOps
import random
import math
import os
from enum import Enum

def log(*args):
    # return
    print(*args)

class Utility:
    @staticmethod
    def randomizeWithinVariance(val,maxVariance=0.2):
        new_val = round(val * (1 + random.uniform(-maxVariance,0)))
        return (new_val)
    
    @staticmethod
    def centerToBoundingBox(center_coords: tuple[int,int],size: tuple[int,int]) -> tuple[int,int,int,int]:
        center_x,center_y = center_coords
        width,height = size
        top_left_x = center_x - width / 2
        top_left_y = center_y - height / 2
        bot_right_x = center_x + width / 2
        bot_right_y = center_y + height /2
        return tuple(map(lambda x: round(x),(top_left_x,top_left_y,bot_right_x,bot_right_y)))

    @staticmethod
    def boundingBoxToCenter(box_coords : tuple[int,int,int,int]) -> tuple[int,int]:
        top_left_x,top_left_y,*_ = box_coords
        center_x = round(top_left_x + SyntheticGenerator.EGG_RADIUS)
        center_y = round(top_left_y + SyntheticGenerator.EGG_RADIUS)
        return (center_x,center_y)
    
    @staticmethod
    def getRandomCoordinate(center: tuple[int,int], max_radius: int):
        random_radius = random.randrange(0, max_radius)
        random_angle = random.random() * math.pi * 2
        center_x, center_y = center
        random_coord = (center_x + random_radius * math.sin(random_angle), center_y + random_radius * math.cos(random_angle))
        return Utility.centerToBoundingBox(random_coord,(SyntheticGenerator.EGG_RADIUS*2,SyntheticGenerator.EGG_RADIUS * 2))

    @staticmethod
    def translateBoxInPolarCoords(curr_bounds: tuple[int,int,int,int], deltaR: float|int, dir: float):
        top_left_x,top_left_y,bot_right_x,bot_right_y = curr_bounds
        log(f'translating from:: {curr_bounds}, deltaR: {deltaR}, dir: {dir}')
        next_top_left_x = deltaR * math.sin(dir) + top_left_x
        next_top_left_y = deltaR * math.cos(dir) + top_left_y
        next_bot_right_x = deltaR * math.sin(dir) + bot_right_x
        next_bot_right_y = deltaR * math.cos(dir) + bot_right_y
        next_bounds = tuple(map(lambda x: round(x),(next_top_left_x,next_top_left_y,next_bot_right_x,next_bot_right_y)))
        log(f'raw translated:: {(next_top_left_x,next_top_left_y,next_bot_right_x,next_bot_right_y)}')
        log(f'translated:: {next_bounds}')
        return next_bounds

class EggType(Enum):
    VIABLE = 'viable'
    NON_VIABLE = 'non_viable'

class SyntheticEgg:
    def __init__(self,image,type : EggType):
        self.image = image
        self.type = type

class SyntheticGeneratorResult:
    def __init__(self,image,total_count,viable_count):
        self.image = image
        self.total_count = total_count
        self.viable_count = viable_count

class SyntheticGenerator:
    INVALID_LOCATION = (-1,-1,-1,-1)
    EGG_RADIUS = 14
    EGG_SPACING = 5
    OVERLAP_PIXEL_THRESHOLD = 50000
    RANDOM_LOCATION_RETRIES_LIMIT = 300
    def __init__(self,viable_images,non_viable_images,backgrounds):
        self.image_height = 600
        self.image_width = 600
        self.boundary_radius = 250
        self.separation_chance = 0
        self.egg_fill = 0.6
        self.viables = list(map(lambda im: im.resize((SyntheticGenerator.EGG_RADIUS * 2,SyntheticGenerator.EGG_RADIUS * 2)).convert('RGBA'),viable_images))
        self.non_viables = list(map(lambda im: im.resize((SyntheticGenerator.EGG_RADIUS * 2,SyntheticGenerator.EGG_RADIUS * 2)).convert('RGBA'),non_viable_images))
        self.backgrounds = list(map(lambda im: im.resize((self.image_width,self.image_height)).convert('RGBA'),backgrounds))

    def _getNextEgg(self,viable_percent:float) -> SyntheticEgg:
        r = random.random()
        if r < viable_percent:
            return SyntheticEgg(random.choice(self.viables),EggType.VIABLE)
        else:
            return SyntheticEgg(random.choice(self.non_viables),EggType.NON_VIABLE)

    def _isCoordinateEmpty(self,curr_location,pixels):
        next_location_pixels = self._calculateLocationPixels(pixels,(curr_location[0],curr_location[1]))
        next_location_pixels_sum = sum(map(lambda tup: sum(tup),next_location_pixels))
        return next_location_pixels_sum < SyntheticGenerator.OVERLAP_PIXEL_THRESHOLD

    def _isCoordinateWithinBounds(self,curr_location):
        top_left_x,top_left_y,bot_right_x,bot_right_y = curr_location
        overflows_horizontal = top_left_x < 0 or bot_right_x < 0 or top_left_x > self.image_width - SyntheticGenerator.EGG_RADIUS or bot_right_x > self.image_width - SyntheticGenerator.EGG_RADIUS
        overflows_vertical = top_left_y < 0 or bot_right_y < 0 or top_left_y > self.image_height - SyntheticGenerator.EGG_RADIUS or bot_right_y > self.image_height - SyntheticGenerator.EGG_RADIUS

        curr_center = Utility.boundingBoxToCenter(curr_location)
        image_center = (self.image_width/2,self.image_height/2)
        overflows_bounding_circle = math.dist(curr_center,image_center) > self.boundary_radius
        return not (overflows_horizontal or overflows_vertical or overflows_bounding_circle)

    def _getNextSpawnLocation(self,curr_location,pixels):
        next_location = SyntheticGenerator.INVALID_LOCATION
        is_next_location_empty = False
        rand_num = random.random()
        angle_choices = [math.pi/4, 2*math.pi/4, 3 * math.pi/4, 4* math.pi/4, 5*math.pi/4, 6 * math.pi / 4, 7 * math.pi/4, 8 * math.pi /4]
        if curr_location == SyntheticGenerator.INVALID_LOCATION or rand_num < self.separation_chance:
            next_location = Utility.getRandomCoordinate((round(self.image_width/2),round(self.image_height/2)),round(self.image_width/2 - SyntheticGenerator.EGG_RADIUS * 2))
        else:
            random.shuffle(angle_choices)
            next_angle = angle_choices.pop()
            next_location = Utility.translateBoxInPolarCoords(curr_location,SyntheticGenerator.EGG_RADIUS * 2 + SyntheticGenerator.EGG_SPACING,next_angle)

        is_next_location_empty = self._isCoordinateEmpty(next_location,pixels)
        is_next_location_within_bounds = self._isCoordinateWithinBounds(next_location)
        retry_count = 0
        while not (is_next_location_empty and is_next_location_within_bounds) and retry_count < SyntheticGenerator.RANDOM_LOCATION_RETRIES_LIMIT:
            if len(angle_choices) == 0:
                next_location = Utility.getRandomCoordinate((round(self.image_width/2),round(self.image_height/2)),round(self.image_width/2 - SyntheticGenerator.EGG_RADIUS * 2))
            else:
                next_location = Utility.translateBoxInPolarCoords(curr_location,SyntheticGenerator.EGG_RADIUS * 2 + SyntheticGenerator.EGG_SPACING,angle_choices.pop())
            is_next_location_empty = self._isCoordinateEmpty(next_location,pixels)
            is_next_location_within_bounds = self._isCoordinateWithinBounds(next_location)
            retry_count+= 1

        return next_location if is_next_location_empty and is_next_location_within_bounds else SyntheticGenerator.INVALID_LOCATION

    def _calculateLocationPixels(self,pixels,curr_location_top):
        top_offset = self.image_width * curr_location_top[1]
        left_offset = curr_location_top[0]
        location_pixels = []
    
        for _ in range(self.EGG_RADIUS * 2):
            location_pixels.extend(pixels[top_offset+left_offset:top_offset+left_offset+self.EGG_RADIUS*2])
            top_offset += self.image_width
        return location_pixels

    def generateEggImages(self,number_images=10,min_eggs=50,max_eggs=100,viable_percent=0.5):
        for i in range(number_images):
            im = Image.new('RGBA',(self.image_width,self.image_height))
            backgroundIm = random.choice(self.backgrounds).copy()
            next_spawn_location = SyntheticGenerator.INVALID_LOCATION
            egg_count = 0
            viable_count = 0
            animation_frames = []
            for _ in range(random.randint(min_eggs,max_eggs)):
                next_egg = self._getNextEgg(viable_percent)
                pixels = list(im.getdata())
                next_spawn_location = self._getNextSpawnLocation(next_spawn_location,pixels)
                if next_spawn_location == SyntheticGenerator.INVALID_LOCATION: break
                log(f'VALID TOP: {next_spawn_location[0],next_spawn_location[1]} BOTTOM: {next_spawn_location[2], next_spawn_location[3]}')
                log(f'Image size: {im.size[0]}x{im.size[1]}')
                im.paste(next_egg.image,next_spawn_location,next_egg.image)
                if next_egg.type == EggType.VIABLE:
                    viable_count+= 1
                # animation_frames.append(im.copy())
                egg_count+=1
            log(f'Egg count: {egg_count}')  
            # animation_frames[0].save('animation.gif',save_all=True,append_images=animation_frames[1:],duration=2)
            backgroundIm.paste(im,mask=im)
            yield SyntheticGeneratorResult(backgroundIm,egg_count,viable_count)

 # Dev test
random.seed(1234)
viable_image_dir = 'fertilized'
non_viable_image_dir = 'unfertilized'
backgrounds_dir = 'background'

viable_images = [Image.open(os.path.join(viable_image_dir,img)) for img in os.listdir(viable_image_dir) if img.endswith('png')]
non_viable_images = [Image.open(os.path.join(non_viable_image_dir,img)) for img in os.listdir(non_viable_image_dir) if img.endswith('png')]
backgrounds = [Image.open(os.path.join(backgrounds_dir,img)) for img in os.listdir(backgrounds_dir) if img.endswith('png')]

frog_generator = SyntheticGenerator(viable_images,non_viable_images,backgrounds)

image_num = 0
for viable_percent in range(20,100,20):
    for result in frog_generator.generateEggImages(number_images=2,min_eggs=100,max_eggs=150,viable_percent=viable_percent/100):
        result.image.save(f'{image_num}-synthetic-eggs-{result.viable_count}-{result.total_count-result.viable_count}.png')
        image_num+=1
