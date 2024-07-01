from PIL import Image
import random
import math
import os
from enum import Enum
import sys
from datetime import datetime


def log(*args):
    # return
    print(*args)


class Utility:
    @staticmethod
    def randomizeWithinVariance(val: int | float, maxVariance=0.2) -> int | float:
        new_val = round(val * (1 + random.uniform(-maxVariance, 0)))
        return new_val

    @staticmethod
    def centerToBoundingBox(
        center_coords: tuple[int, int], size: tuple[int, int]
    ) -> tuple[int, int, int, int]:
        center_x, center_y = center_coords
        width, height = size
        top_left_x = round(center_x - width / 2)
        top_left_y = round(center_y - height / 2)
        bot_right_x = round(center_x + width / 2)
        bot_right_y = round(center_y + height / 2)
        return (top_left_x, top_left_y, bot_right_x, bot_right_y)

    @staticmethod
    def boundingBoxToCenter(
        box_coords: tuple[int, int, int, int], box_size: tuple[int, int]
    ) -> tuple[int, int]:
        top_left_x, top_left_y, *_ = box_coords
        center_x = round(top_left_x + box_size[0] / 2)
        center_y = round(top_left_y + box_size[1] / 2)
        return (center_x, center_y)

    @staticmethod
    def getRandomCoordinate(
        center: tuple[int, int], max_radius: int
    ) -> tuple[int, int]:
        random_radius = random.randrange(0, max_radius)
        random_angle = random.random() * math.pi * 2
        center_x, center_y = center
        random_coord = (
            round(center_x + random_radius * math.sin(random_angle)),
            round(center_y + random_radius * math.cos(random_angle)),
        )
        return random_coord

    @staticmethod
    def translateBoxInPolarCoords(
        curr_bounds: tuple[int, int, int, int], deltaR: float | int, dir: float
    ):
        top_left_x, top_left_y, bot_right_x, bot_right_y = curr_bounds
        next_top_left_x = deltaR * math.sin(dir) + top_left_x
        next_top_left_y = deltaR * math.cos(dir) + top_left_y
        next_bot_right_x = deltaR * math.sin(dir) + bot_right_x
        next_bot_right_y = deltaR * math.cos(dir) + bot_right_y
        next_bounds = tuple(
            map(
                lambda x: round(x),
                (next_top_left_x, next_top_left_y, next_bot_right_x, next_bot_right_y),
            )
        )
        return next_bounds


class EggType(Enum):
    VIABLE = 0
    NON_VIABLE = 1


class SyntheticEgg:
    def __init__(self, image: Image.Image, type: EggType):
        self.image = image
        self.type = type


class SyntheticGeneratorResult:
    def __init__(self, image: Image.Image, total_count: int, viable_count: int):
        self.image = image
        self.total_count = total_count
        self.viable_count = viable_count


class SyntheticGenerator:
    INVALID_LOCATION = (-1, -1, -1, -1)
    EGG_RADIUS = 12
    EGG_SPACING = 3
    OVERLAP_PIXEL_THRESHOLD = 50000
    RANDOM_LOCATION_RETRIES_LIMIT = 300

    def __init__(
        self,
        viable_images_dir: str,
        non_viable_images_dir: str,
        backgrounds_dir: str,
        separation_chance: float = 0,
    ):
        self.image_height = 600
        self.image_width = 600
        self.boundary_radius = 250
        self.separation_chance = separation_chance
        self.viables = [
            Image.open(os.path.join(viable_images_dir, img))
            .resize(
                (SyntheticGenerator.EGG_RADIUS * 2, SyntheticGenerator.EGG_RADIUS * 2)
            )
            .convert("RGBA")
            for img in os.listdir(viable_images_dir)
            if img.endswith("png")
        ]
        self.non_viables = [
            Image.open(os.path.join(non_viable_images_dir, img))
            .resize(
                (SyntheticGenerator.EGG_RADIUS * 2, SyntheticGenerator.EGG_RADIUS * 2)
            )
            .convert("RGBA")
            for img in os.listdir(non_viable_images_dir)
            if img.endswith("png")
        ]
        self.backgrounds = [
            Image.open(os.path.join(backgrounds_dir, img))
            .resize((self.image_width, self.image_height))
            .convert("RGBA")
            for img in os.listdir(backgrounds_dir)
            if img.endswith("png")
        ]

    def _getNextEgg(self, viable_percent: float) -> SyntheticEgg:
        r = random.random()
        if r < viable_percent:
            return SyntheticEgg(random.choice(self.viables), EggType.VIABLE)
        else:
            return SyntheticEgg(random.choice(self.non_viables), EggType.NON_VIABLE)

    def _isCoordinateEmpty(self, curr_location, pixels):
        next_location_pixels = self._calculateLocationPixels(
            pixels, (curr_location[0], curr_location[1])
        )
        next_location_pixels_sum = sum(map(lambda tup: sum(tup), next_location_pixels))
        return next_location_pixels_sum < SyntheticGenerator.OVERLAP_PIXEL_THRESHOLD

    def _isCoordinateWithinBounds(self, curr_location):
        top_left_x, top_left_y, bot_right_x, bot_right_y = curr_location
        overflows_horizontal = (
            top_left_x < 0
            or bot_right_x < 0
            or top_left_x > self.image_width - SyntheticGenerator.EGG_RADIUS
            or bot_right_x > self.image_width - SyntheticGenerator.EGG_RADIUS
        )
        overflows_vertical = (
            top_left_y < 0
            or bot_right_y < 0
            or top_left_y > self.image_height - SyntheticGenerator.EGG_RADIUS
            or bot_right_y > self.image_height - SyntheticGenerator.EGG_RADIUS
        )

        curr_center = Utility.boundingBoxToCenter(
            curr_location,
            box_size=(
                SyntheticGenerator.EGG_RADIUS * 2,
                SyntheticGenerator.EGG_RADIUS * 2,
            ),
        )
        image_center = (self.image_width / 2, self.image_height / 2)
        overflows_bounding_circle = (
            math.dist(curr_center, image_center) > self.boundary_radius
        )
        return not (
            overflows_horizontal or overflows_vertical or overflows_bounding_circle
        )

    def _getNextSpawnLocation(
        self, curr_location: tuple[int, int, int, int], pixels: list[int]
    ):
        next_location = SyntheticGenerator.INVALID_LOCATION
        is_next_location_empty = False
        rand_num = random.random()
        angle_choices = [i * math.pi / 4 for i in range(1, 9)]
        if (
            curr_location == SyntheticGenerator.INVALID_LOCATION
            or rand_num < self.separation_chance
        ):
            next_location_center = Utility.getRandomCoordinate(
                (round(self.image_width / 2), round(self.image_height / 2)),
                self.boundary_radius,
            )
            next_location = Utility.centerToBoundingBox(
                next_location_center,
                (SyntheticGenerator.EGG_RADIUS * 2, SyntheticGenerator.EGG_RADIUS * 2),
            )
        else:
            random.shuffle(angle_choices)
            next_angle = angle_choices.pop()
            next_location = Utility.translateBoxInPolarCoords(
                curr_location,
                SyntheticGenerator.EGG_RADIUS * 2 + SyntheticGenerator.EGG_SPACING,
                next_angle,
            )

        is_next_location_empty = self._isCoordinateEmpty(next_location, pixels)
        is_next_location_within_bounds = self._isCoordinateWithinBounds(next_location)
        retry_count = 0
        while (
            not (is_next_location_empty and is_next_location_within_bounds)
            and retry_count < SyntheticGenerator.RANDOM_LOCATION_RETRIES_LIMIT
        ):
            if len(angle_choices) == 0:
                next_location_center = Utility.getRandomCoordinate(
                    (round(self.image_width / 2), round(self.image_height / 2)),
                    self.boundary_radius,
                )
                next_location = Utility.centerToBoundingBox(
                    next_location_center,
                    (
                        SyntheticGenerator.EGG_RADIUS * 2,
                        SyntheticGenerator.EGG_RADIUS * 2,
                    ),
                )
            else:
                next_location = Utility.translateBoxInPolarCoords(
                    curr_location,
                    SyntheticGenerator.EGG_RADIUS * 2 + SyntheticGenerator.EGG_SPACING,
                    angle_choices.pop(),
                )
            is_next_location_empty = self._isCoordinateEmpty(next_location, pixels)
            is_next_location_within_bounds = self._isCoordinateWithinBounds(
                next_location
            )
            retry_count += 1

        return (
            next_location
            if is_next_location_empty and is_next_location_within_bounds
            else SyntheticGenerator.INVALID_LOCATION
        )

    def _calculateLocationPixels(self, pixels, curr_location_top):
        top_offset = self.image_width * curr_location_top[1]
        left_offset = curr_location_top[0]
        location_pixels = []

        for _ in range(self.EGG_RADIUS * 2):
            location_pixels.extend(
                pixels[
                    top_offset
                    + left_offset : top_offset
                    + left_offset
                    + self.EGG_RADIUS * 2
                ]
            )
            top_offset += self.image_width
        return location_pixels

    def generete(
        self,
        number_images: int = 10,
        min_eggs: int = 50,
        max_eggs: int = 100,
        viable_percent: float = 0.5,
        save_dir: str = "generated",
    ) -> None:
        image_count = 0

        if not os.path.isdir(save_dir):
            # exist_ok=True handles race conditions when run in multiple threads
            os.makedirs(save_dir, exist_ok=True)

        for i in range(number_images):
            im = Image.new("RGBA", (self.image_width, self.image_height))
            backgroundIm = random.choice(self.backgrounds).copy()
            next_spawn_location = SyntheticGenerator.INVALID_LOCATION
            egg_count = 0
            viable_count = 0
            annotations = (
                []
            )  # class,normalized_center_x,normalized_center_y,egg_normalized_width,egg_normalized_height
            while egg_count < random.randint(min_eggs, max_eggs) and (
                egg_count == 0
                or next_spawn_location != SyntheticGenerator.INVALID_LOCATION
            ):
                next_egg = self._getNextEgg(viable_percent)
                pixels = list(im.getdata())
                next_spawn_location = self._getNextSpawnLocation(
                    next_spawn_location, pixels
                )
                next_spawn_center = Utility.boundingBoxToCenter(next_spawn_location,(SyntheticGenerator.EGG_RADIUS*2,SyntheticGenerator.EGG_RADIUS*2))
                im.paste(next_egg.image, next_spawn_location, next_egg.image)
                if next_egg.type == EggType.VIABLE:
                    viable_count += 1
                egg_count += 1
                annotations.append(f'{next_egg.type.value} {next_spawn_center[0]/self.image_width:.4f} {next_spawn_center[1]/self.image_height:.4f} {SyntheticGenerator.EGG_RADIUS*2/self.image_width} {SyntheticGenerator.EGG_RADIUS*2/self.image_height}\n')
            backgroundIm.paste(im, mask=im)
            backgroundIm.save(
                f"{save_dir}/{image_count}-synthetic-eggs-{viable_count}-{egg_count-viable_count}.png"
            )
            with open(f"{save_dir}/{image_count}-synthetic-eggs-{viable_count}-{egg_count-viable_count}.txt",'w+') as f:
                f.writelines(annotations)
            image_count += 1


def parseCmdArgs(args: list[str]):
    validArgs = dict(
        [
            ("number_images", int),
            ("min_eggs", int),
            ("max_eggs", int),
            ("viable_percent", float),
            ("save_dir", str),
        ]
    )

    parsed_args = dict()
    for arg in args:
        if not arg.startswith("--"):
            raise Exception(
                "Options have the format --option_name=value. Invalid option: %s" % arg
            )
        trimmed_arg = arg.lstrip("-")
        option, value = trimmed_arg.split("=")
        if option not in validArgs:
            raise Exception("Invalid option: %s" % option)
        try:
            parsed_args[option] = validArgs[option](value)
        except ValueError:
            raise Exception("Value for option --%s is invalid" % option)

    return parsed_args


def main():
    cmd_args = sys.argv[1:]
    # Dev test
    viable_images_dir = "fertilized_images"
    non_viable_images_dir = "unfertilized_images"
    backgrounds_dir = "background_images"

    frog_generator = SyntheticGenerator(
        viable_images_dir, non_viable_images_dir, backgrounds_dir
    )

    default_args = dict(
        number_images=2,
        min_eggs=10,
        max_eggs=20,
        viable_percent=0.6,
        save_dir="generated",
    )

    cmd_options_parsed = default_args
    try:
        cmd_options_parsed.update(parseCmdArgs(cmd_args))
        log("[%s] Generating images with options:" % datetime.now(), cmd_options_parsed)
        frog_generator.generete(**cmd_options_parsed)
        log(
            "[%s] Finished generating images with options:" % datetime.now(),
            cmd_options_parsed,
        )
    except Exception as e:
        log("\n")
        if hasattr(e, "message"):
            log(e.message)
        else:
            log(e)
        log("\n")
    return 0


if __name__ == "__main__":
    main()
