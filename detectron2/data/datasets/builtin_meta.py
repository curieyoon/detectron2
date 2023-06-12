# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

"""
Note:
For your custom dataset, there is no need to hard-code metadata anywhere in the code.
For example, for COCO-format dataset, metadata will be obtained automatically
when calling `load_coco_json`. For other dataset, metadata may also be obtained in other ways
during loading.

However, we hard-coded metadata for a few common dataset here.
The only goal is to allow users who don't have these dataset to use pre-trained models.
Users don't have to download a COCO json (which contains metadata), in order to visualize a
COCO model (with correct class names and colors).
"""


# All coco categories, together with their nice-looking visualization colors
# It's from https://github.com/cocodataset/panopticapi/blob/master/panoptic_coco_categories.json

# 338 OpenImages-Test classes with segmentation masks
COCO_CATEGORIES = [
    {"color": [1, 2, 3], "isthing": 1, "id": 1, "name": "Tortoise"},
    {"color": [1, 2, 3], "isthing": 1, "id": 3, "name": "Magpie"},
    {"color": [1, 2, 3], "isthing": 1, "id": 4, "name": "Sea turtle"},
    {"color": [1, 2, 3], "isthing": 1, "id": 5, "name": "Football"},
    {"color": [1, 2, 3], "isthing": 1, "id": 6, "name": "Ambulance"},
    {"color": [1, 2, 3], "isthing": 1, "id": 11, "name": "Toy"},
    {"color": [1, 2, 3], "isthing": 1, "id": 14, "name": "Apple"},
    {"color": [1, 2, 3], "isthing": 1, "id": 19, "name": "Beer"},
    {"color": [1, 2, 3], "isthing": 1, "id": 20, "name": "Chopsticks"},
    {"color": [1, 2, 3], "isthing": 1, "id": 22, "name": "Bird"},
    {"color": [1, 2, 3], "isthing": 1, "id": 24, "name": "Traffic light"},
    {"color": [1, 2, 3], "isthing": 1, "id": 25, "name": "Croissant"},
    {"color": [1, 2, 3], "isthing": 1, "id": 26, "name": "Cucumber"},
    {"color": [1, 2, 3], "isthing": 1, "id": 27, "name": "Radish"},
    {"color": [1, 2, 3], "isthing": 1, "id": 28, "name": "Towel"},
    {"color": [1, 2, 3], "isthing": 1, "id": 30, "name": "Skull"},
    {"color": [1, 2, 3], "isthing": 1, "id": 31, "name": "Washing machine"},
    {"color": [1, 2, 3], "isthing": 1, "id": 32, "name": "Glove"},
    {"color": [1, 2, 3], "isthing": 1, "id": 34, "name": "Belt"},
    {"color": [1, 2, 3], "isthing": 1, "id": 38, "name": "Ball"},
    {"color": [1, 2, 3], "isthing": 1, "id": 39, "name": "Backpack"},
    {"color": [1, 2, 3], "isthing": 1, "id": 44, "name": "Surfboard"},
    {"color": [1, 2, 3], "isthing": 1, "id": 45, "name": "Boot"},
    {"color": [1, 2, 3], "isthing": 1, "id": 47, "name": "Hot dog"},
    {"color": [1, 2, 3], "isthing": 1, "id": 48, "name": "Shorts"},
    {"color": [1, 2, 3], "isthing": 1, "id": 50, "name": "Bus"},
    {"color": [1, 2, 3], "isthing": 1, "id": 51, "name": "Boy"},
    {"color": [1, 2, 3], "isthing": 1, "id": 52, "name": "Screwdriver"},
    {"color": [1, 2, 3], "isthing": 1, "id": 53, "name": "Bicycle wheel"},
    {"color": [1, 2, 3], "isthing": 1, "id": 54, "name": "Barge"},
    {"color": [1, 2, 3], "isthing": 1, "id": 55, "name": "Laptop"},
    {"color": [1, 2, 3], "isthing": 1, "id": 56, "name": "Miniskirt"},
    {"color": [1, 2, 3], "isthing": 1, "id": 57, "name": "Drill (Tool)"},
    {"color": [1, 2, 3], "isthing": 1, "id": 58, "name": "Dress"},
    {"color": [1, 2, 3], "isthing": 1, "id": 59, "name": "Bear"},
    {"color": [1, 2, 3], "isthing": 1, "id": 60, "name": "Waffle"},
    {"color": [1, 2, 3], "isthing": 1, "id": 61, "name": "Pancake"},
    {"color": [1, 2, 3], "isthing": 1, "id": 62, "name": "Brown bear"},
    {"color": [1, 2, 3], "isthing": 1, "id": 63, "name": "Woodpecker"},
    {"color": [1, 2, 3], "isthing": 1, "id": 64, "name": "Blue jay"},
    {"color": [1, 2, 3], "isthing": 1, "id": 65, "name": "Pretzel"},
    {"color": [1, 2, 3], "isthing": 1, "id": 66, "name": "Bagel"},
    {"color": [1, 2, 3], "isthing": 1, "id": 68, "name": "Teapot"},
    {"color": [1, 2, 3], "isthing": 1, "id": 69, "name": "Person"},
    {"color": [1, 2, 3], "isthing": 1, "id": 71, "name": "Swimwear"},
    {"color": [1, 2, 3], "isthing": 1, "id": 75, "name": "Bat (Animal)"},
    {"color": [1, 2, 3], "isthing": 1, "id": 76, "name": "Starfish"},
    {"color": [1, 2, 3], "isthing": 1, "id": 77, "name": "Popcorn"},
    {"color": [1, 2, 3], "isthing": 1, "id": 78, "name": "Burrito"},
    {"color": [1, 2, 3], "isthing": 1, "id": 80, "name": "Balloon"},
    {"color": [1, 2, 3], "isthing": 1, "id": 81, "name": "Wrench"},
    {"color": [1, 2, 3], "isthing": 1, "id": 83, "name": "Vehicle registration plate"},
    {"color": [1, 2, 3], "isthing": 1, "id": 85, "name": "Toaster"},
    {"color": [1, 2, 3], "isthing": 1, "id": 86, "name": "Flashlight"},
    {"color": [1, 2, 3], "isthing": 1, "id": 89, "name": "Limousine"},
    {"color": [1, 2, 3], "isthing": 1, "id": 91, "name": "Carnivore"},
    {"color": [1, 2, 3], "isthing": 1, "id": 92, "name": "Scissors"},
    {"color": [1, 2, 3], "isthing": 1, "id": 94, "name": "Computer keyboard"},
    {"color": [1, 2, 3], "isthing": 1, "id": 95, "name": "Printer"},
    {"color": [1, 2, 3], "isthing": 1, "id": 96, "name": "Traffic sign"},
    {"color": [1, 2, 3], "isthing": 1, "id": 98, "name": "Shirt"},
    {"color": [1, 2, 3], "isthing": 1, "id": 100, "name": "Cheese"},
    {"color": [1, 2, 3], "isthing": 1, "id": 101, "name": "Sock"},
    {"color": [1, 2, 3], "isthing": 1, "id": 102, "name": "Fire hydrant"},
    {"color": [1, 2, 3], "isthing": 1, "id": 105, "name": "Tie"},
    {"color": [1, 2, 3], "isthing": 1, "id": 108, "name": "Suitcase"},
    {"color": [1, 2, 3], "isthing": 1, "id": 109, "name": "Muffin"},
    {"color": [1, 2, 3], "isthing": 1, "id": 112, "name": "Snowmobile"},
    {"color": [1, 2, 3], "isthing": 1, "id": 113, "name": "Clock"},
    {"color": [1, 2, 3], "isthing": 1, "id": 115, "name": "Cattle"},
    {"color": [1, 2, 3], "isthing": 1, "id": 116, "name": "Cello"},
    {"color": [1, 2, 3], "isthing": 1, "id": 117, "name": "Jet ski"},
    {"color": [1, 2, 3], "isthing": 1, "id": 118, "name": "Camel"},
    {"color": [1, 2, 3], "isthing": 1, "id": 120, "name": "Suit"},
    {"color": [1, 2, 3], "isthing": 1, "id": 122, "name": "Cat"},
    {"color": [1, 2, 3], "isthing": 1, "id": 123, "name": "Bronze sculpture"},
    {"color": [1, 2, 3], "isthing": 1, "id": 124, "name": "Juice"},
    {"color": [1, 2, 3], "isthing": 1, "id": 128, "name": "Computer mouse"},
    {"color": [1, 2, 3], "isthing": 1, "id": 129, "name": "Cookie"},
    {"color": [1, 2, 3], "isthing": 1, "id": 132, "name": "Coin"},
    {"color": [1, 2, 3], "isthing": 1, "id": 133, "name": "Calculator"},
    {"color": [1, 2, 3], "isthing": 1, "id": 134, "name": "Cocktail"},
    {"color": [1, 2, 3], "isthing": 1, "id": 136, "name": "Box"},
    {"color": [1, 2, 3], "isthing": 1, "id": 137, "name": "Stapler"},
    {"color": [1, 2, 3], "isthing": 1, "id": 138, "name": "Christmas tree"},
    {"color": [1, 2, 3], "isthing": 1, "id": 139, "name": "Cowboy hat"},
    {"color": [1, 2, 3], "isthing": 1, "id": 141, "name": "Studio couch"},
    {"color": [1, 2, 3], "isthing": 1, "id": 145, "name": "Drink"},
    {"color": [1, 2, 3], "isthing": 1, "id": 146, "name": "Zucchini"},
    {"color": [1, 2, 3], "isthing": 1, "id": 147, "name": "Ladle"},
    {"color": [1, 2, 3], "isthing": 1, "id": 148, "name": "Human mouth"},
    {"color": [1, 2, 3], "isthing": 1, "id": 150, "name": "Dice"},
    {"color": [1, 2, 3], "isthing": 1, "id": 151, "name": "Oven"},
    {"color": [1, 2, 3], "isthing": 1, "id": 154, "name": "Couch"},
    {"color": [1, 2, 3], "isthing": 1, "id": 155, "name": "Cricket ball"},
    {"color": [1, 2, 3], "isthing": 1, "id": 156, "name": "Winter melon"},
    {"color": [1, 2, 3], "isthing": 1, "id": 157, "name": "Spatula"},
    {"color": [1, 2, 3], "isthing": 1, "id": 158, "name": "Whiteboard"},
    {"color": [1, 2, 3], "isthing": 1, "id": 161, "name": "Hat"},
    {"color": [1, 2, 3], "isthing": 1, "id": 162, "name": "Shower"},
    {"color": [1, 2, 3], "isthing": 1, "id": 163, "name": "Eraser"},
    {"color": [1, 2, 3], "isthing": 1, "id": 164, "name": "Fedora"},
    {"color": [1, 2, 3], "isthing": 1, "id": 165, "name": "Guacamole"},
    {"color": [1, 2, 3], "isthing": 1, "id": 166, "name": "Dagger"},
    {"color": [1, 2, 3], "isthing": 1, "id": 167, "name": "Scarf"},
    {"color": [1, 2, 3], "isthing": 1, "id": 168, "name": "Dolphin"},
    {"color": [1, 2, 3], "isthing": 1, "id": 169, "name": "Sombrero"},
    {"color": [1, 2, 3], "isthing": 1, "id": 171, "name": "Mug"},
    {"color": [1, 2, 3], "isthing": 1, "id": 172, "name": "Tap"},
    {"color": [1, 2, 3], "isthing": 1, "id": 173, "name": "Harbor seal"},
    {"color": [1, 2, 3], "isthing": 1, "id": 177, "name": "Human body"},
    {"color": [1, 2, 3], "isthing": 1, "id": 178, "name": "Roller skates"},
    {"color": [1, 2, 3], "isthing": 1, "id": 179, "name": "Coffee cup"},
    {"color": [1, 2, 3], "isthing": 1, "id": 183, "name": "Stop sign"},
    {"color": [1, 2, 3], "isthing": 1, "id": 185, "name": "Volleyball (Ball)"},
    {"color": [1, 2, 3], "isthing": 1, "id": 186, "name": "Vase"},
    {"color": [1, 2, 3], "isthing": 1, "id": 187, "name": "Slow cooker"},
    {"color": [1, 2, 3], "isthing": 1, "id": 189, "name": "Coffee"},
    {"color": [1, 2, 3], "isthing": 1, "id": 191, "name": "Paper towel"},
    {"color": [1, 2, 3], "isthing": 1, "id": 194, "name": "Sun hat"},
    {"color": [1, 2, 3], "isthing": 1, "id": 196, "name": "Flying disc"},
    {"color": [1, 2, 3], "isthing": 1, "id": 197, "name": "Skirt"},
    {"color": [1, 2, 3], "isthing": 1, "id": 206, "name": "Barrel"},
    {"color": [1, 2, 3], "isthing": 1, "id": 207, "name": "Kite"},
    {"color": [1, 2, 3], "isthing": 1, "id": 208, "name": "Tart"},
    {"color": [1, 2, 3], "isthing": 1, "id": 210, "name": "Fox"},
    {"color": [1, 2, 3], "isthing": 1, "id": 211, "name": "Flag"},
    {"color": [1, 2, 3], "isthing": 1, "id": 219, "name": "Guitar"},
    {"color": [1, 2, 3], "isthing": 1, "id": 220, "name": "Pillow"},
    {"color": [1, 2, 3], "isthing": 1, "id": 223, "name": "Grape"},
    {"color": [1, 2, 3], "isthing": 1, "id": 224, "name": "Human ear"},
    {"color": [1, 2, 3], "isthing": 1, "id": 225, "name": "Power plugs and sockets"},
    {"color": [1, 2, 3], "isthing": 1, "id": 226, "name": "Panda"},
    {"color": [1, 2, 3], "isthing": 1, "id": 227, "name": "Giraffe"},
    {"color": [1, 2, 3], "isthing": 1, "id": 228, "name": "Woman"},
    {"color": [1, 2, 3], "isthing": 1, "id": 229, "name": "Door handle"},
    {"color": [1, 2, 3], "isthing": 1, "id": 230, "name": "Rhinoceros"},
    {"color": [1, 2, 3], "isthing": 1, "id": 232, "name": "Goldfish"},
    {"color": [1, 2, 3], "isthing": 1, "id": 234, "name": "Goat"},
    {"color": [1, 2, 3], "isthing": 1, "id": 235, "name": "Baseball bat"},
    {"color": [1, 2, 3], "isthing": 1, "id": 236, "name": "Baseball glove"},
    {"color": [1, 2, 3], "isthing": 1, "id": 237, "name": "Mixing bowl"},
    {"color": [1, 2, 3], "isthing": 1, "id": 240, "name": "Light switch"},
    {"color": [1, 2, 3], "isthing": 1, "id": 242, "name": "Horse"},
    {"color": [1, 2, 3], "isthing": 1, "id": 246, "name": "Sofa bed"},
    {"color": [1, 2, 3], "isthing": 1, "id": 247, "name": "Adhesive tape"},
    {"color": [1, 2, 3], "isthing": 1, "id": 251, "name": "Saucer"},
    {"color": [1, 2, 3], "isthing": 1, "id": 252, "name": "Harpsichord"},
    {"color": [1, 2, 3], "isthing": 1, "id": 254, "name": "Heater"},
    {"color": [1, 2, 3], "isthing": 1, "id": 255, "name": "Harmonica"},
    {"color": [1, 2, 3], "isthing": 1, "id": 256, "name": "Hamster"},
    {"color": [1, 2, 3], "isthing": 1, "id": 259, "name": "Kettle"},
    {"color": [1, 2, 3], "isthing": 1, "id": 262, "name": "Drinking straw"},
    {"color": [1, 2, 3], "isthing": 1, "id": 264, "name": "Hair dryer"},
    {"color": [1, 2, 3], "isthing": 1, "id": 268, "name": "Food processor"},
    {"color": [1, 2, 3], "isthing": 1, "id": 272, "name": "Punching bag"},
    {"color": [1, 2, 3], "isthing": 1, "id": 273, "name": "Common fig"},
    {"color": [1, 2, 3], "isthing": 1, "id": 275, "name": "Jaguar (Animal)"},
    {"color": [1, 2, 3], "isthing": 1, "id": 276, "name": "Golf ball"},
    {"color": [1, 2, 3], "isthing": 1, "id": 278, "name": "Alarm clock"},
    {"color": [1, 2, 3], "isthing": 1, "id": 279, "name": "Filing cabinet"},
    {"color": [1, 2, 3], "isthing": 1, "id": 280, "name": "Artichoke"},
    {"color": [1, 2, 3], "isthing": 1, "id": 283, "name": "Kangaroo"},
    {"color": [1, 2, 3], "isthing": 1, "id": 284, "name": "Koala"},
    {"color": [1, 2, 3], "isthing": 1, "id": 285, "name": "Knife"},
    {"color": [1, 2, 3], "isthing": 1, "id": 286, "name": "Bottle"},
    {"color": [1, 2, 3], "isthing": 1, "id": 287, "name": "Bottle opener"},
    {"color": [1, 2, 3], "isthing": 1, "id": 288, "name": "Lynx"},
    {"color": [1, 2, 3], "isthing": 1, "id": 290, "name": "Lighthouse"},
    {"color": [1, 2, 3], "isthing": 1, "id": 291, "name": "Dumbbell"},
    {"color": [1, 2, 3], "isthing": 1, "id": 293, "name": "Bowl"},
    {"color": [1, 2, 3], "isthing": 1, "id": 296, "name": "Lizard"},
    {"color": [1, 2, 3], "isthing": 1, "id": 297, "name": "Billiard table"},
    {"color": [1, 2, 3], "isthing": 1, "id": 299, "name": "Mouse"},
    {"color": [1, 2, 3], "isthing": 1, "id": 300, "name": "Motorcycle"},
    {"color": [1, 2, 3], "isthing": 1, "id": 302, "name": "Swim cap"},
    {"color": [1, 2, 3], "isthing": 1, "id": 303, "name": "Frying pan"},
    {"color": [1, 2, 3], "isthing": 1, "id": 306, "name": "Missile"},
    {"color": [1, 2, 3], "isthing": 1, "id": 307, "name": "Bust"},
    {"color": [1, 2, 3], "isthing": 1, "id": 308, "name": "Man"},
    {"color": [1, 2, 3], "isthing": 1, "id": 310, "name": "Milk"},
    {"color": [1, 2, 3], "isthing": 1, "id": 313, "name": "Mobile phone"},
    {"color": [1, 2, 3], "isthing": 1, "id": 315, "name": "Mushroom"},
    {"color": [1, 2, 3], "isthing": 1, "id": 317, "name": "Pitcher (Container)"},
    {"color": [1, 2, 3], "isthing": 1, "id": 320, "name": "Table tennis racket"},
    {"color": [1, 2, 3], "isthing": 1, "id": 321, "name": "Pencil case"},
    {"color": [1, 2, 3], "isthing": 1, "id": 324, "name": "Briefcase"},
    {"color": [1, 2, 3], "isthing": 1, "id": 325, "name": "Kitchen knife"},
    {"color": [1, 2, 3], "isthing": 1, "id": 326, "name": "Nail (Construction)"},
    {"color": [1, 2, 3], "isthing": 1, "id": 327, "name": "Tennis ball"},
    {"color": [1, 2, 3], "isthing": 1, "id": 328, "name": "Plastic bag"},
    {"color": [1, 2, 3], "isthing": 1, "id": 330, "name": "Chest of drawers"},
    {"color": [1, 2, 3], "isthing": 1, "id": 331, "name": "Ostrich"},
    {"color": [1, 2, 3], "isthing": 1, "id": 332, "name": "Piano"},
    {"color": [1, 2, 3], "isthing": 1, "id": 333, "name": "Girl"},
    {"color": [1, 2, 3], "isthing": 1, "id": 335, "name": "Potato"},
    {"color": [1, 2, 3], "isthing": 1, "id": 339, "name": "Penguin"},
    {"color": [1, 2, 3], "isthing": 1, "id": 340, "name": "Pumpkin"},
    {"color": [1, 2, 3], "isthing": 1, "id": 341, "name": "Pear"},
    {"color": [1, 2, 3], "isthing": 1, "id": 343, "name": "Polar bear"},
    {"color": [1, 2, 3], "isthing": 1, "id": 347, "name": "Pizza"},
    {"color": [1, 2, 3], "isthing": 1, "id": 348, "name": "Digital clock"},
    {"color": [1, 2, 3], "isthing": 1, "id": 349, "name": "Pig"},
    {"color": [1, 2, 3], "isthing": 1, "id": 350, "name": "Reptile"},
    {"color": [1, 2, 3], "isthing": 1, "id": 352, "name": "Lipstick"},
    {"color": [1, 2, 3], "isthing": 1, "id": 353, "name": "Skateboard"},
    {"color": [1, 2, 3], "isthing": 1, "id": 354, "name": "Raven"},
    {"color": [1, 2, 3], "isthing": 1, "id": 355, "name": "High heels"},
    {"color": [1, 2, 3], "isthing": 1, "id": 356, "name": "Red panda"},
    {"color": [1, 2, 3], "isthing": 1, "id": 357, "name": "Rose"},
    {"color": [1, 2, 3], "isthing": 1, "id": 358, "name": "Rabbit"},
    {"color": [1, 2, 3], "isthing": 1, "id": 359, "name": "Sculpture"},
    {"color": [1, 2, 3], "isthing": 1, "id": 360, "name": "Saxophone"},
    {"color": [1, 2, 3], "isthing": 1, "id": 363, "name": "Submarine sandwich"},
    {"color": [1, 2, 3], "isthing": 1, "id": 365, "name": "Sword"},
    {"color": [1, 2, 3], "isthing": 1, "id": 366, "name": "Picture frame"},
    {"color": [1, 2, 3], "isthing": 1, "id": 368, "name": "Loveseat"},
    {"color": [1, 2, 3], "isthing": 1, "id": 370, "name": "Squirrel"},
    {"color": [1, 2, 3], "isthing": 1, "id": 375, "name": "Segway"},
    {"color": [1, 2, 3], "isthing": 1, "id": 377, "name": "Snake"},
    {"color": [1, 2, 3], "isthing": 1, "id": 379, "name": "Skyscraper"},
    {"color": [1, 2, 3], "isthing": 1, "id": 380, "name": "Sheep"},
    {"color": [1, 2, 3], "isthing": 1, "id": 383, "name": "Tea"},
    {"color": [1, 2, 3], "isthing": 1, "id": 384, "name": "Tank"},
    {"color": [1, 2, 3], "isthing": 1, "id": 387, "name": "Torch"},
    {"color": [1, 2, 3], "isthing": 1, "id": 388, "name": "Tiger"},
    {"color": [1, 2, 3], "isthing": 1, "id": 389, "name": "Strawberry"},
    {"color": [1, 2, 3], "isthing": 1, "id": 392, "name": "Tomato"},
    {"color": [1, 2, 3], "isthing": 1, "id": 393, "name": "Train"},
    {"color": [1, 2, 3], "isthing": 1, "id": 397, "name": "Trousers"},
    {"color": [1, 2, 3], "isthing": 1, "id": 400, "name": "Truck"},
    {"color": [1, 2, 3], "isthing": 1, "id": 405, "name": "Handbag"},
    {"color": [1, 2, 3], "isthing": 1, "id": 407, "name": "Wine"},
    {"color": [1, 2, 3], "isthing": 1, "id": 409, "name": "Wheel"},
    {"color": [1, 2, 3], "isthing": 1, "id": 411, "name": "Wok"},
    {"color": [1, 2, 3], "isthing": 1, "id": 412, "name": "Whale"},
    {"color": [1, 2, 3], "isthing": 1, "id": 413, "name": "Zebra"},
    {"color": [1, 2, 3], "isthing": 1, "id": 415, "name": "Jug"},
    {"color": [1, 2, 3], "isthing": 1, "id": 418, "name": "Monkey"},
    {"color": [1, 2, 3], "isthing": 1, "id": 419, "name": "Lion"},
    {"color": [1, 2, 3], "isthing": 1, "id": 420, "name": "Bread"},
    {"color": [1, 2, 3], "isthing": 1, "id": 421, "name": "Platter"},
    {"color": [1, 2, 3], "isthing": 1, "id": 422, "name": "Chicken"},
    {"color": [1, 2, 3], "isthing": 1, "id": 423, "name": "Eagle"},
    {"color": [1, 2, 3], "isthing": 1, "id": 425, "name": "Owl"},
    {"color": [1, 2, 3], "isthing": 1, "id": 426, "name": "Duck"},
    {"color": [1, 2, 3], "isthing": 1, "id": 427, "name": "Turtle"},
    {"color": [1, 2, 3], "isthing": 1, "id": 428, "name": "Hippopotamus"},
    {"color": [1, 2, 3], "isthing": 1, "id": 429, "name": "Crocodile"},
    {"color": [1, 2, 3], "isthing": 1, "id": 430, "name": "Toilet"},
    {"color": [1, 2, 3], "isthing": 1, "id": 431, "name": "Toilet paper"},
    {"color": [1, 2, 3], "isthing": 1, "id": 433, "name": "Clothing"},
    {"color": [1, 2, 3], "isthing": 1, "id": 435, "name": "Lemon"},
    {"color": [1, 2, 3], "isthing": 1, "id": 438, "name": "Frog"},
    {"color": [1, 2, 3], "isthing": 1, "id": 439, "name": "Banana"},
    {"color": [1, 2, 3], "isthing": 1, "id": 440, "name": "Rocket"},
    {"color": [1, 2, 3], "isthing": 1, "id": 443, "name": "Tablet computer"},
    {"color": [1, 2, 3], "isthing": 1, "id": 444, "name": "Waste container"},
    {"color": [1, 2, 3], "isthing": 1, "id": 446, "name": "Dog"},
    {"color": [1, 2, 3], "isthing": 1, "id": 447, "name": "Book"},
    {"color": [1, 2, 3], "isthing": 1, "id": 448, "name": "Elephant"},
    {"color": [1, 2, 3], "isthing": 1, "id": 449, "name": "Shark"},
    {"color": [1, 2, 3], "isthing": 1, "id": 450, "name": "Candle"},
    {"color": [1, 2, 3], "isthing": 1, "id": 451, "name": "Leopard"},
    {"color": [1, 2, 3], "isthing": 1, "id": 452, "name": "Axe"},
    {"color": [1, 2, 3], "isthing": 1, "id": 453, "name": "Hand dryer"},
    {"color": [1, 2, 3], "isthing": 1, "id": 454, "name": "Soap dispenser"},
    {"color": [1, 2, 3], "isthing": 1, "id": 456, "name": "Flower"},
    {"color": [1, 2, 3], "isthing": 1, "id": 457, "name": "Canary"},
    {"color": [1, 2, 3], "isthing": 1, "id": 458, "name": "Cheetah"},
    {"color": [1, 2, 3], "isthing": 1, "id": 460, "name": "Hamburger"},
    {"color": [1, 2, 3], "isthing": 1, "id": 463, "name": "Fish"},
    {"color": [1, 2, 3], "isthing": 1, "id": 465, "name": "Garden Asparagus"},
    {"color": [1, 2, 3], "isthing": 1, "id": 467, "name": "Hedgehog"},
    {"color": [1, 2, 3], "isthing": 1, "id": 468, "name": "Airplane"},
    {"color": [1, 2, 3], "isthing": 1, "id": 469, "name": "Spoon"},
    {"color": [1, 2, 3], "isthing": 1, "id": 470, "name": "Otter"},
    {"color": [1, 2, 3], "isthing": 1, "id": 471, "name": "Bull"},
    {"color": [1, 2, 3], "isthing": 1, "id": 472, "name": "Oyster"},
    {"color": [1, 2, 3], "isthing": 1, "id": 481, "name": "Orange"},
    {"color": [1, 2, 3], "isthing": 1, "id": 483, "name": "Beaker"},
    {"color": [1, 2, 3], "isthing": 1, "id": 489, "name": "Goose"},
    {"color": [1, 2, 3], "isthing": 1, "id": 490, "name": "Mule"},
    {"color": [1, 2, 3], "isthing": 1, "id": 491, "name": "Swan"},
    {"color": [1, 2, 3], "isthing": 1, "id": 492, "name": "Peach"},
    {"color": [1, 2, 3], "isthing": 1, "id": 494, "name": "Seat belt"},
    {"color": [1, 2, 3], "isthing": 1, "id": 495, "name": "Raccoon"},
    {"color": [1, 2, 3], "isthing": 1, "id": 499, "name": "Camera"},
    {"color": [1, 2, 3], "isthing": 1, "id": 500, "name": "Squash (Plant)"},
    {"color": [1, 2, 3], "isthing": 1, "id": 501, "name": "Racket"},
    {"color": [1, 2, 3], "isthing": 1, "id": 505, "name": "Diaper"},
    {"color": [1, 2, 3], "isthing": 1, "id": 507, "name": "Falcon"},
    {"color": [1, 2, 3], "isthing": 1, "id": 511, "name": "Cabbage"},
    {"color": [1, 2, 3], "isthing": 1, "id": 512, "name": "Carrot"},
    {"color": [1, 2, 3], "isthing": 1, "id": 513, "name": "Mango"},
    {"color": [1, 2, 3], "isthing": 1, "id": 514, "name": "Jeans"},
    {"color": [1, 2, 3], "isthing": 1, "id": 515, "name": "Flowerpot"},
    {"color": [1, 2, 3], "isthing": 1, "id": 519, "name": "Envelope"},
    {"color": [1, 2, 3], "isthing": 1, "id": 520, "name": "Cake"},
    {"color": [1, 2, 3], "isthing": 1, "id": 522, "name": "Common sunflower"},
    {"color": [1, 2, 3], "isthing": 1, "id": 523, "name": "Microwave oven"},
    {"color": [1, 2, 3], "isthing": 1, "id": 526, "name": "Sea lion"},
    {"color": [1, 2, 3], "isthing": 1, "id": 529, "name": "Watch"},
    {"color": [1, 2, 3], "isthing": 1, "id": 532, "name": "Parrot"},
    {"color": [1, 2, 3], "isthing": 1, "id": 533, "name": "Handgun"},
    {"color": [1, 2, 3], "isthing": 1, "id": 534, "name": "Sparrow"},
    {"color": [1, 2, 3], "isthing": 1, "id": 535, "name": "Van"},
    {"color": [1, 2, 3], "isthing": 1, "id": 539, "name": "Corded phone"},
    {"color": [1, 2, 3], "isthing": 1, "id": 541, "name": "Tennis racket"},
    {"color": [1, 2, 3], "isthing": 1, "id": 545, "name": "Dog bed"},
    {"color": [1, 2, 3], "isthing": 1, "id": 549, "name": "Facial tissue holder"},
    {"color": [1, 2, 3], "isthing": 1, "id": 553, "name": "Ruler"},
    {"color": [1, 2, 3], "isthing": 1, "id": 554, "name": "Luggage and bags"},
    {"color": [1, 2, 3], "isthing": 1, "id": 556, "name": "Broccoli"},
    {"color": [1, 2, 3], "isthing": 1, "id": 558, "name": "Pastry"},
    {"color": [1, 2, 3], "isthing": 1, "id": 559, "name": "Grapefruit"},
    {"color": [1, 2, 3], "isthing": 1, "id": 560, "name": "Band-aid"},
    {"color": [1, 2, 3], "isthing": 1, "id": 562, "name": "Bell pepper"},
    {"color": [1, 2, 3], "isthing": 1, "id": 563, "name": "Turkey"},
    {"color": [1, 2, 3], "isthing": 1, "id": 565, "name": "Pomegranate"},
    {"color": [1, 2, 3], "isthing": 1, "id": 566, "name": "Doughnut"},
    {"color": [1, 2, 3], "isthing": 1, "id": 569, "name": "Pen"},
    {"color": [1, 2, 3], "isthing": 1, "id": 571, "name": "Car"},
    {"color": [1, 2, 3], "isthing": 1, "id": 572, "name": "Aircraft"},
    {"color": [1, 2, 3], "isthing": 1, "id": 574, "name": "Skunk"},
    {"color": [1, 2, 3], "isthing": 1, "id": 575, "name": "Teddy bear"},
    {"color": [1, 2, 3], "isthing": 1, "id": 576, "name": "Watermelon"},
    {"color": [1, 2, 3], "isthing": 1, "id": 577, "name": "Cantaloupe"},
    {"color": [1, 2, 3], "isthing": 1, "id": 579, "name": "Flute"},
    {"color": [1, 2, 3], "isthing": 1, "id": 580, "name": "Balance beam"},
    {"color": [1, 2, 3], "isthing": 1, "id": 581, "name": "Sandwich"},
    {"color": [1, 2, 3], "isthing": 1, "id": 584, "name": "Binoculars"},
    {"color": [1, 2, 3], "isthing": 1, "id": 586, "name": "Ipod"},
    {"color": [1, 2, 3], "isthing": 1, "id": 593, "name": "Alpaca"},
    {"color": [1, 2, 3], "isthing": 1, "id": 594, "name": "Taxi"},
    {"color": [1, 2, 3], "isthing": 1, "id": 595, "name": "Canoe"},
    {"color": [1, 2, 3], "isthing": 1, "id": 596, "name": "Remote control"},
    {"color": [1, 2, 3], "isthing": 1, "id": 598, "name": "Rugby ball"},


]


# COCO_CATEGORIES = [
#     {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"},
#     {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "bicycle"},
#     {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "car"},
#     {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "motorcycle"},
#     {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "airplane"},
#     {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "bus"},
#     {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "train"},
#     {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "truck"},
#     {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "boat"},
#     {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "traffic light"},
#     {"color": [100, 170, 30], "isthing": 1, "id": 11, "name": "fire hydrant"},
#     {"color": [220, 220, 0], "isthing": 1, "id": 13, "name": "stop sign"},
#     {"color": [175, 116, 175], "isthing": 1, "id": 14, "name": "parking meter"},
#     {"color": [250, 0, 30], "isthing": 1, "id": 15, "name": "bench"},
#     {"color": [165, 42, 42], "isthing": 1, "id": 16, "name": "bird"},
#     {"color": [255, 77, 255], "isthing": 1, "id": 17, "name": "cat"},
#     {"color": [0, 226, 252], "isthing": 1, "id": 18, "name": "dog"},
#     {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "horse"},
#     {"color": [0, 82, 0], "isthing": 1, "id": 20, "name": "sheep"},
#     {"color": [120, 166, 157], "isthing": 1, "id": 21, "name": "cow"},
#     {"color": [110, 76, 0], "isthing": 1, "id": 22, "name": "elephant"},
#     {"color": [174, 57, 255], "isthing": 1, "id": 23, "name": "bear"},
#     {"color": [199, 100, 0], "isthing": 1, "id": 24, "name": "zebra"},
#     {"color": [72, 0, 118], "isthing": 1, "id": 25, "name": "giraffe"},
#     {"color": [255, 179, 240], "isthing": 1, "id": 27, "name": "backpack"},
#     {"color": [0, 125, 92], "isthing": 1, "id": 28, "name": "umbrella"},
#     {"color": [209, 0, 151], "isthing": 1, "id": 31, "name": "handbag"},
#     {"color": [188, 208, 182], "isthing": 1, "id": 32, "name": "tie"},
#     {"color": [0, 220, 176], "isthing": 1, "id": 33, "name": "suitcase"},
#     {"color": [255, 99, 164], "isthing": 1, "id": 34, "name": "frisbee"},
#     {"color": [92, 0, 73], "isthing": 1, "id": 35, "name": "skis"},
#     {"color": [133, 129, 255], "isthing": 1, "id": 36, "name": "snowboard"},
#     {"color": [78, 180, 255], "isthing": 1, "id": 37, "name": "sports ball"},
#     {"color": [0, 228, 0], "isthing": 1, "id": 38, "name": "kite"},
#     {"color": [174, 255, 243], "isthing": 1, "id": 39, "name": "baseball bat"},
#     {"color": [45, 89, 255], "isthing": 1, "id": 40, "name": "baseball glove"},
#     {"color": [134, 134, 103], "isthing": 1, "id": 41, "name": "skateboard"},
#     {"color": [145, 148, 174], "isthing": 1, "id": 42, "name": "surfboard"},
#     {"color": [255, 208, 186], "isthing": 1, "id": 43, "name": "tennis racket"},
#     {"color": [197, 226, 255], "isthing": 1, "id": 44, "name": "bottle"},
#     {"color": [171, 134, 1], "isthing": 1, "id": 46, "name": "wine glass"},
#     {"color": [109, 63, 54], "isthing": 1, "id": 47, "name": "cup"},
#     {"color": [207, 138, 255], "isthing": 1, "id": 48, "name": "fork"},
#     {"color": [151, 0, 95], "isthing": 1, "id": 49, "name": "knife"},
#     {"color": [9, 80, 61], "isthing": 1, "id": 50, "name": "spoon"},
#     {"color": [84, 105, 51], "isthing": 1, "id": 51, "name": "bowl"},
#     {"color": [74, 65, 105], "isthing": 1, "id": 52, "name": "banana"},
#     {"color": [166, 196, 102], "isthing": 1, "id": 53, "name": "apple"},
#     {"color": [208, 195, 210], "isthing": 1, "id": 54, "name": "sandwich"},
#     {"color": [255, 109, 65], "isthing": 1, "id": 55, "name": "orange"},
#     {"color": [0, 143, 149], "isthing": 1, "id": 56, "name": "broccoli"},
#     {"color": [179, 0, 194], "isthing": 1, "id": 57, "name": "carrot"},
#     {"color": [209, 99, 106], "isthing": 1, "id": 58, "name": "hot dog"},
#     {"color": [5, 121, 0], "isthing": 1, "id": 59, "name": "pizza"},
#     {"color": [227, 255, 205], "isthing": 1, "id": 60, "name": "donut"},
#     {"color": [147, 186, 208], "isthing": 1, "id": 61, "name": "cake"},
#     {"color": [153, 69, 1], "isthing": 1, "id": 62, "name": "chair"},
#     {"color": [3, 95, 161], "isthing": 1, "id": 63, "name": "couch"},
#     {"color": [163, 255, 0], "isthing": 1, "id": 64, "name": "potted plant"},
#     {"color": [119, 0, 170], "isthing": 1, "id": 65, "name": "bed"},
#     {"color": [0, 182, 199], "isthing": 1, "id": 67, "name": "dining table"},
#     {"color": [0, 165, 120], "isthing": 1, "id": 70, "name": "toilet"},
#     {"color": [183, 130, 88], "isthing": 1, "id": 72, "name": "tv"},
#     {"color": [95, 32, 0], "isthing": 1, "id": 73, "name": "laptop"},
#     {"color": [130, 114, 135], "isthing": 1, "id": 74, "name": "mouse"},
#     {"color": [110, 129, 133], "isthing": 1, "id": 75, "name": "remote"},
#     {"color": [166, 74, 118], "isthing": 1, "id": 76, "name": "keyboard"},
#     {"color": [219, 142, 185], "isthing": 1, "id": 77, "name": "cell phone"},
#     {"color": [79, 210, 114], "isthing": 1, "id": 78, "name": "microwave"},
#     {"color": [178, 90, 62], "isthing": 1, "id": 79, "name": "oven"},
#     {"color": [65, 70, 15], "isthing": 1, "id": 80, "name": "toaster"},
#     {"color": [127, 167, 115], "isthing": 1, "id": 81, "name": "sink"},
#     {"color": [59, 105, 106], "isthing": 1, "id": 82, "name": "refrigerator"},
#     {"color": [142, 108, 45], "isthing": 1, "id": 84, "name": "book"},
#     {"color": [196, 172, 0], "isthing": 1, "id": 85, "name": "clock"},
#     {"color": [95, 54, 80], "isthing": 1, "id": 86, "name": "vase"},
#     {"color": [128, 76, 255], "isthing": 1, "id": 87, "name": "scissors"},
#     {"color": [201, 57, 1], "isthing": 1, "id": 88, "name": "teddy bear"},
#     {"color": [246, 0, 122], "isthing": 1, "id": 89, "name": "hair drier"},
#     {"color": [191, 162, 208], "isthing": 1, "id": 90, "name": "toothbrush"},
#     {"color": [255, 255, 128], "isthing": 0, "id": 92, "name": "banner"},
#     {"color": [147, 211, 203], "isthing": 0, "id": 93, "name": "blanket"},
#     {"color": [150, 100, 100], "isthing": 0, "id": 95, "name": "bridge"},
#     {"color": [168, 171, 172], "isthing": 0, "id": 100, "name": "cardboard"},
#     {"color": [146, 112, 198], "isthing": 0, "id": 107, "name": "counter"},
#     {"color": [210, 170, 100], "isthing": 0, "id": 109, "name": "curtain"},
#     {"color": [92, 136, 89], "isthing": 0, "id": 112, "name": "door-stuff"},
#     {"color": [218, 88, 184], "isthing": 0, "id": 118, "name": "floor-wood"},
#     {"color": [241, 129, 0], "isthing": 0, "id": 119, "name": "flower"},
#     {"color": [217, 17, 255], "isthing": 0, "id": 122, "name": "fruit"},
#     {"color": [124, 74, 181], "isthing": 0, "id": 125, "name": "gravel"},
#     {"color": [70, 70, 70], "isthing": 0, "id": 128, "name": "house"},
#     {"color": [255, 228, 255], "isthing": 0, "id": 130, "name": "light"},
#     {"color": [154, 208, 0], "isthing": 0, "id": 133, "name": "mirror-stuff"},
#     {"color": [193, 0, 92], "isthing": 0, "id": 138, "name": "net"},
#     {"color": [76, 91, 113], "isthing": 0, "id": 141, "name": "pillow"},
#     {"color": [255, 180, 195], "isthing": 0, "id": 144, "name": "platform"},
#     {"color": [106, 154, 176], "isthing": 0, "id": 145, "name": "playingfield"},
#     {"color": [230, 150, 140], "isthing": 0, "id": 147, "name": "railroad"},
#     {"color": [60, 143, 255], "isthing": 0, "id": 148, "name": "river"},
#     {"color": [128, 64, 128], "isthing": 0, "id": 149, "name": "road"},
#     {"color": [92, 82, 55], "isthing": 0, "id": 151, "name": "roof"},
#     {"color": [254, 212, 124], "isthing": 0, "id": 154, "name": "sand"},
#     {"color": [73, 77, 174], "isthing": 0, "id": 155, "name": "sea"},
#     {"color": [255, 160, 98], "isthing": 0, "id": 156, "name": "shelf"},
#     {"color": [255, 255, 255], "isthing": 0, "id": 159, "name": "snow"},
#     {"color": [104, 84, 109], "isthing": 0, "id": 161, "name": "stairs"},
#     {"color": [169, 164, 131], "isthing": 0, "id": 166, "name": "tent"},
#     {"color": [225, 199, 255], "isthing": 0, "id": 168, "name": "towel"},
#     {"color": [137, 54, 74], "isthing": 0, "id": 171, "name": "wall-brick"},
#     {"color": [135, 158, 223], "isthing": 0, "id": 175, "name": "wall-stone"},
#     {"color": [7, 246, 231], "isthing": 0, "id": 176, "name": "wall-tile"},
#     {"color": [107, 255, 200], "isthing": 0, "id": 177, "name": "wall-wood"},
#     {"color": [58, 41, 149], "isthing": 0, "id": 178, "name": "water-other"},
#     {"color": [183, 121, 142], "isthing": 0, "id": 180, "name": "window-blind"},
#     {"color": [255, 73, 97], "isthing": 0, "id": 181, "name": "window-other"},
#     {"color": [107, 142, 35], "isthing": 0, "id": 184, "name": "tree-merged"},
#     {"color": [190, 153, 153], "isthing": 0, "id": 185, "name": "fence-merged"},
#     {"color": [146, 139, 141], "isthing": 0, "id": 186, "name": "ceiling-merged"},
#     {"color": [70, 130, 180], "isthing": 0, "id": 187, "name": "sky-other-merged"},
#     {"color": [134, 199, 156], "isthing": 0, "id": 188, "name": "cabinet-merged"},
#     {"color": [209, 226, 140], "isthing": 0, "id": 189, "name": "table-merged"},
#     {"color": [96, 36, 108], "isthing": 0, "id": 190, "name": "floor-other-merged"},
#     {"color": [96, 96, 96], "isthing": 0, "id": 191, "name": "pavement-merged"},
#     {"color": [64, 170, 64], "isthing": 0, "id": 192, "name": "mountain-merged"},
#     {"color": [152, 251, 152], "isthing": 0, "id": 193, "name": "grass-merged"},
#     {"color": [208, 229, 228], "isthing": 0, "id": 194, "name": "dirt-merged"},
#     {"color": [206, 186, 171], "isthing": 0, "id": 195, "name": "paper-merged"},
#     {"color": [152, 161, 64], "isthing": 0, "id": 196, "name": "food-other-merged"},
#     {"color": [116, 112, 0], "isthing": 0, "id": 197, "name": "building-other-merged"},
#     {"color": [0, 114, 143], "isthing": 0, "id": 198, "name": "rock-merged"},
#     {"color": [102, 102, 156], "isthing": 0, "id": 199, "name": "wall-other-merged"},
#     {"color": [250, 141, 255], "isthing": 0, "id": 200, "name": "rug-merged"},
# ]


# fmt: off
COCO_PERSON_KEYPOINT_NAMES = (
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)
# fmt: on

# Pairs of keypoints that should be exchanged under horizontal flipping
COCO_PERSON_KEYPOINT_FLIP_MAP = (
    ("left_eye", "right_eye"),
    ("left_ear", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
)

# rules for pairs of keypoints to draw a line between, and the line color to use.
KEYPOINT_CONNECTION_RULES = [
    # face
    ("left_ear", "left_eye", (102, 204, 255)),
    ("right_ear", "right_eye", (51, 153, 255)),
    ("left_eye", "nose", (102, 0, 204)),
    ("nose", "right_eye", (51, 102, 255)),
    # upper-body
    ("left_shoulder", "right_shoulder", (255, 128, 0)),
    ("left_shoulder", "left_elbow", (153, 255, 204)),
    ("right_shoulder", "right_elbow", (128, 229, 255)),
    ("left_elbow", "left_wrist", (153, 255, 153)),
    ("right_elbow", "right_wrist", (102, 255, 224)),
    # lower-body
    ("left_hip", "right_hip", (255, 102, 0)),
    ("left_hip", "left_knee", (255, 255, 77)),
    ("right_hip", "right_knee", (153, 255, 204)),
    ("left_knee", "left_ankle", (191, 255, 128)),
    ("right_knee", "right_ankle", (255, 195, 77)),
]

# All Cityscapes categories, together with their nice-looking visualization colors
# It's from https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py  # noqa
CITYSCAPES_CATEGORIES = [
    {"color": (128, 64, 128), "isthing": 0, "id": 7, "trainId": 0, "name": "road"},
    {"color": (244, 35, 232), "isthing": 0, "id": 8, "trainId": 1, "name": "sidewalk"},
    {"color": (70, 70, 70), "isthing": 0, "id": 11, "trainId": 2, "name": "building"},
    {"color": (102, 102, 156), "isthing": 0, "id": 12, "trainId": 3, "name": "wall"},
    {"color": (190, 153, 153), "isthing": 0, "id": 13, "trainId": 4, "name": "fence"},
    {"color": (153, 153, 153), "isthing": 0, "id": 17, "trainId": 5, "name": "pole"},
    {"color": (250, 170, 30), "isthing": 0, "id": 19, "trainId": 6, "name": "traffic light"},
    {"color": (220, 220, 0), "isthing": 0, "id": 20, "trainId": 7, "name": "traffic sign"},
    {"color": (107, 142, 35), "isthing": 0, "id": 21, "trainId": 8, "name": "vegetation"},
    {"color": (152, 251, 152), "isthing": 0, "id": 22, "trainId": 9, "name": "terrain"},
    {"color": (70, 130, 180), "isthing": 0, "id": 23, "trainId": 10, "name": "sky"},
    {"color": (220, 20, 60), "isthing": 1, "id": 24, "trainId": 11, "name": "person"},
    {"color": (255, 0, 0), "isthing": 1, "id": 25, "trainId": 12, "name": "rider"},
    {"color": (0, 0, 142), "isthing": 1, "id": 26, "trainId": 13, "name": "car"},
    {"color": (0, 0, 70), "isthing": 1, "id": 27, "trainId": 14, "name": "truck"},
    {"color": (0, 60, 100), "isthing": 1, "id": 28, "trainId": 15, "name": "bus"},
    {"color": (0, 80, 100), "isthing": 1, "id": 31, "trainId": 16, "name": "train"},
    {"color": (0, 0, 230), "isthing": 1, "id": 32, "trainId": 17, "name": "motorcycle"},
    {"color": (119, 11, 32), "isthing": 1, "id": 33, "trainId": 18, "name": "bicycle"},
]

# fmt: off
ADE20K_SEM_SEG_CATEGORIES = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road, route", "bed", "window ", "grass", "cabinet", "sidewalk, pavement", "person", "earth, ground", "door", "table", "mountain, mount", "plant", "curtain", "chair", "car", "water", "painting, picture", "sofa", "shelf", "house", "sea", "mirror", "rug", "field", "armchair", "seat", "fence", "desk", "rock, stone", "wardrobe, closet, press", "lamp", "tub", "rail", "cushion", "base, pedestal, stand", "box", "column, pillar", "signboard, sign", "chest of drawers, chest, bureau, dresser", "counter", "sand", "sink", "skyscraper", "fireplace", "refrigerator, icebox", "grandstand, covered stand", "path", "stairs", "runway", "case, display case, showcase, vitrine", "pool table, billiard table, snooker table", "pillow", "screen door, screen", "stairway, staircase", "river", "bridge, span", "bookcase", "blind, screen", "coffee table", "toilet, can, commode, crapper, pot, potty, stool, throne", "flower", "book", "hill", "bench", "countertop", "stove", "palm, palm tree", "kitchen island", "computer", "swivel chair", "boat", "bar", "arcade machine", "hovel, hut, hutch, shack, shanty", "bus", "towel", "light", "truck", "tower", "chandelier", "awning, sunshade, sunblind", "street lamp", "booth", "tv", "plane", "dirt track", "clothes", "pole", "land, ground, soil", "bannister, banister, balustrade, balusters, handrail", "escalator, moving staircase, moving stairway", "ottoman, pouf, pouffe, puff, hassock", "bottle", "buffet, counter, sideboard", "poster, posting, placard, notice, bill, card", "stage", "van", "ship", "fountain", "conveyer belt, conveyor belt, conveyer, conveyor, transporter", "canopy", "washer, automatic washer, washing machine", "plaything, toy", "pool", "stool", "barrel, cask", "basket, handbasket", "falls", "tent", "bag", "minibike, motorbike", "cradle", "oven", "ball", "food, solid food", "step, stair", "tank, storage tank", "trade name", "microwave", "pot", "animal", "bicycle", "lake", "dishwasher", "screen", "blanket, cover", "sculpture", "hood, exhaust hood", "sconce", "vase", "traffic light", "tray", "trash can", "fan", "pier", "crt screen", "plate", "monitor", "bulletin board", "shower", "radiator", "glass, drinking glass", "clock", "flag", # noqa
]
# After processed by `prepare_ade20k_sem_seg.py`, id 255 means ignore
# fmt: on


def _get_coco_instances_meta():
    thing_ids = [k["id"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 338, len(thing_ids)
    # Mapping from the incontiguous COCO category id to an id in [0, 79]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def _get_coco_panoptic_separated_meta():
    """
    Returns metadata for "separated" version of the panoptic segmentation dataset.
    """
    stuff_ids = [k["id"] for k in COCO_CATEGORIES if k["isthing"] == 0]
    assert len(stuff_ids) == 0, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 53], used in models) to ids in the dataset (used for processing results)
    # The id 0 is mapped to an extra category "thing".
    stuff_dataset_id_to_contiguous_id = {k: i + 1 for i, k in enumerate(stuff_ids)}
    # When converting COCO panoptic annotations to semantic annotations
    # We label the "thing" category to 0
    stuff_dataset_id_to_contiguous_id[0] = 0

    # 54 names for COCO stuff categories (including "things")
    stuff_classes = ["things"] + [
        k["name"].replace("-other", "").replace("-merged", "")
        for k in COCO_CATEGORIES
        if k["isthing"] == 0
    ]

    # NOTE: I randomly picked a color for things
    stuff_colors = [[82, 18, 128]] + [k["color"] for k in COCO_CATEGORIES if k["isthing"] == 0]
    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    ret.update(_get_coco_instances_meta())
    return ret


def _get_builtin_metadata(dataset_name):
    if dataset_name == "coco":
        return _get_coco_instances_meta()
    if dataset_name == "coco_panoptic_separated":
        return _get_coco_panoptic_separated_meta()
    elif dataset_name == "coco_panoptic_standard":
        meta = {}
        # The following metadata maps contiguous id from [0, #thing categories +
        # #stuff categories) to their names and colors. We have to replica of the
        # same name and color under "thing_*" and "stuff_*" because the current
        # visualization function in D2 handles thing and class classes differently
        # due to some heuristic used in Panoptic FPN. We keep the same naming to
        # enable reusing existing visualization functions.
        thing_classes = [k["name"] for k in COCO_CATEGORIES]
        thing_colors = [k["color"] for k in COCO_CATEGORIES]
        stuff_classes = [k["name"] for k in COCO_CATEGORIES]
        stuff_colors = [k["color"] for k in COCO_CATEGORIES]

        meta["thing_classes"] = thing_classes
        meta["thing_colors"] = thing_colors
        meta["stuff_classes"] = stuff_classes
        meta["stuff_colors"] = stuff_colors

        # Convert category id for training:
        #   category id: like semantic segmentation, it is the class id for each
        #   pixel. Since there are some classes not used in evaluation, the category
        #   id is not always contiguous and thus we have two set of category ids:
        #       - original category id: category id in the original dataset, mainly
        #           used for evaluation.
        #       - contiguous category id: [0, #classes), in order to train the linear
        #           softmax classifier.
        thing_dataset_id_to_contiguous_id = {}
        stuff_dataset_id_to_contiguous_id = {}

        for i, cat in enumerate(COCO_CATEGORIES):
            if cat["isthing"]:
                thing_dataset_id_to_contiguous_id[cat["id"]] = i
            else:
                stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
        meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

        return meta
    elif dataset_name == "coco_person":
        return {
            "thing_classes": ["person"],
            "keypoint_names": COCO_PERSON_KEYPOINT_NAMES,
            "keypoint_flip_map": COCO_PERSON_KEYPOINT_FLIP_MAP,
            "keypoint_connection_rules": KEYPOINT_CONNECTION_RULES,
        }
    elif dataset_name == "cityscapes":
        # fmt: off
        CITYSCAPES_THING_CLASSES = [
            "person", "rider", "car", "truck",
            "bus", "train", "motorcycle", "bicycle",
        ]
        CITYSCAPES_STUFF_CLASSES = [
            "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
            "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
            "truck", "bus", "train", "motorcycle", "bicycle",
        ]
        # fmt: on
        return {
            "thing_classes": CITYSCAPES_THING_CLASSES,
            "stuff_classes": CITYSCAPES_STUFF_CLASSES,
        }
    raise KeyError("No built-in metadata for dataset {}".format(dataset_name))
