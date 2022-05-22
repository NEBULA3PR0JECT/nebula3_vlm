import clip
import cv2
import numpy as np
import torch
from PIL import Image
import subprocess
import os
import sys
from tqdm import tqdm
from itertools import islice
from pathlib import Path
from vlm.clip_api import CLIP_API
from torchvision.datasets import CIFAR100


class CLIP_OBJECT_DETECTOR:
    def __init__(self):
        # clip_version = "ViT-L/14@336px"
        self.clip_feat_dim = 768
        clip_api = CLIP_API('vit')
        self.model, self.preprocess = clip_api.get_model()
        if torch.cuda.is_available():
            self.model.cuda().eval()
        else:
            self.model.cpu().eval()
        self.img_size = self.model.visual.input_resolution
        print("Loading CLIP")
        self.clip_api = clip_api
        print("Loading places")
        self.place_feats, self.place_texts = self.load_place_feats()
        print("Loading people")
        self.people_feats, self.people_texts = self.load_people_feats()
        # print("Loading objects")
        # self.object_feats, self.object_texts = self.load_object_feats(self.place_texts)
        print("Loading cifar classes")
        self.cifar_feats, self.cifar_texts = self.load_cifar_obj_feats()
        print("Done loading texts")

    def patch_frames_v(self, frame):
        h, w, channels = frame.shape
        half2 = h//2
        top = frame[:half2, :]
        bottom = frame[half2:, :]
        # saving all the images
        # cv2.imwrite() function will save the image 
        # into your pc
        return(top, bottom)
    
    def patch_frames_h(self, frame):
        h, w, channels = frame.shape
        half = w//2
        # this will be the first column
        left_part = frame[:, :half] 

        right_part = frame[:, half:]  

        # [:,half:] means al the rows and all
        # saving all the images
        # cv2.imwrite() function will save the image 
        # into your pc
        return(right_part, left_part)
       

    def get_text_feats(self, in_text, batch_size=64):
        if torch.cuda.is_available():
            text_tokens = clip.tokenize(in_text).cuda()
        else: 
            text_tokens = clip.tokenize(in_text).cpu()
        text_id = 0
        text_feats = np.zeros((len(in_text), self.clip_feat_dim), dtype=np.float32)
        while text_id < len(text_tokens):  # Batched inference.
            batch_size = min(len(in_text) - text_id, batch_size)
            text_batch = text_tokens[text_id:text_id+batch_size]
            with torch.no_grad():
                batch_feats = self.model.encode_text(text_batch).float()
            batch_feats /= batch_feats.norm(dim=-1, keepdim=True)
            batch_feats = np.float32(batch_feats.cpu())
            text_feats[text_id:text_id+batch_size, :] = batch_feats
            text_id += batch_size
        return(text_feats)

    def get_img_feats(self, img):
        img_pil = Image.fromarray(np.uint8(img))
        img_in = self.preprocess(img_pil)[None, ...]
        if torch.cuda.is_available():
            img_in_c = img_in.cuda()
        else:
            img_in_c= img_in.cpu()
        with torch.no_grad():
            img_feats = self.model.encode_image(img_in_c).float()
        img_feats /= img_feats.norm(dim=-1, keepdim=True)
        img_feats = np.float32(img_feats.cpu())
        return(img_feats)

    def get_nn_text(self, raw_texts, text_feats, img_feats, score):
        scores = text_feats @ img_feats.T
        scores = scores.squeeze()
        high_to_low_texts = []
        high_to_low_scores = []
        high_to_low_ids = np.argsort(scores).squeeze()[::-1]
        high_to_low_texts_ = [raw_texts[i] for i in high_to_low_ids]
        high_to_low_scores_ = np.sort(scores).squeeze()[::-1]
        for t, s in zip(high_to_low_texts_, high_to_low_scores_):
            if s > score:
                high_to_low_texts.append(t)
                high_to_low_scores.append(s)
        return(high_to_low_texts, high_to_low_scores)

    def load_place_feats(self):
        # Load scene categories from Places365.
        if not os.path.exists('categories_places365.txt'):
            subprocess.run(["/usr/bin/wget", "https://raw.githubusercontent.com/zhoubolei/places_devkit/master/categories_places365.txt"])
        place_categories = np.loadtxt('categories_places365.txt', dtype=str)
        place_texts = []
        for place in place_categories[:, 0]:
            place = place.split('/')[2:]
            if len(place) > 1:
                place = place[1] + ' ' + place[0]
            else:
                place = place[0]
            place = place.replace('_', ' ')
            place_texts.append(place)
        prmt = [f'a movie frame of the {p}' for p in place_texts]
        place_feats = self.get_text_feats(prmt)
        return(place_feats, place_texts)

    def load_people_feats(self):
        # Load scene categories from Places365.
        people_triplets = np.load('triplets', allow_pickle=True)
        person_texts = []
        for person in people_triplets:
            person_texts.append(person)
        person_feats = self.get_text_feats(person_texts)
        return(person_feats, person_texts)

    def load_cifar_obj_feats(self):
        #cifar100 = CIFAR100(os.path.expanduser("~/.cache"), transform=self.preprocess, download=True)
        imagenet_classes = ["tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray", "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco", "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper", "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander", "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog", "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin", "box turtle", "banded gecko", "green iguana", "Carolina anole", "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard", "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile", "American alligator", "triceratops", "worm snake", "ring-necked snake", "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake", "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra", "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake", "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider", "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider", "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl", "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck", "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby", "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch", "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab", "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab", "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron", "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot", "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher", "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion", "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel", "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle", "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound", "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound", "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound", "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier", "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier", "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier", "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier", "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer", "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier", "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier", "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever", "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla", "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel", "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel", "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard", "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie", "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann", "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog", "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff", "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky", "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog", "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon", "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle", "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf", "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox", "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat", "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger", "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose", "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle", "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper", "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper", "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly", "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly", "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit", "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse", "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison", "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)", "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat", "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque", "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin", "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey", "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda", "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish", "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown", "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance", "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle", "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo", "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel", "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel", "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)", "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini", "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet", "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra", "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest", "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe", "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton", "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran", "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw", "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking", "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker", "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard", "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot", "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed", "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer", "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table", "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig", "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar", "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder", "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute", "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed", "freight car", "French horn", "frying pan", "fur coat", "garbage truck", "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola", "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine", "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer", "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet", "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar", "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep", "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat", "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library", "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion", "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag", "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask", "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone", "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile", "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor", "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa", "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail", "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina", "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart", "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush", "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench", "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case", "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube", "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball", "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag", "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho", "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug", "printer", "prison", "missile", "projector", "hockey puck", "punching bag", "purse", "quill", "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel", "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator", "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser", "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal", "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard", "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store", "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap", "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door", "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock", "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater", "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight", "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf", "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa", "submarine", "suit", "sundial", "sunglasses", "sunglasses", "sunscreen", "suspension bridge", "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe", "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball", "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof", "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store", "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard", "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling", "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball", "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink", "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle", "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing", "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website", "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu", "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette", "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli", "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber", "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange", "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate", "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito", "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef", "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player", "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn", "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom", "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"]
        prmt = [f"Photo of a {label}" for label in imagenet_classes]
        cifar_feats = self.get_text_feats(prmt)
        return(cifar_feats, imagenet_classes)

    def load_object_feats(self, place_texts):
        if not os.path.exists('dictionary_and_semantic_hierarchy.txt'):
            subprocess.run(["/usr/bin/wget", "https://raw.githubusercontent.com/Tencent/tencent-ml-images/master/data/dictionary_and_semantic_hierarchy.txt"])
        with open('dictionary_and_semantic_hierarchy.txt') as fid:
            object_categories = fid.readlines()
        object_texts = []
        for object_text in object_categories[1:]:
            object_text = object_text.strip()
            object_text = object_text.split('\t')[3]
            safe_list = ''
            for variant in object_text.split(','):
                text = variant.strip()
            if len(text) > 0:
                object_texts.append(text)
        object_texts = [o for o in list(set(object_texts)) if o not in place_texts]  # Remove redundant categories.
        #print(object_texts)
        prmt = [f'Photo of a {o}' for o in object_texts]
        #prmt = [f'This is a {o} in the photo' for o in object_texts]
        object_feats = self.get_text_feats(prmt)
        return(object_feats, object_texts)

    def mdf_selection(self, frame):
        img_feats = self.get_img_feats(frame)
        frame_texts = ['a low quality blurry image', 'a high quality sharp image']
        frame_feats = self.get_text_feats([f'{p}.' for p in frame_texts])
        sorted_frame_texts, frame_scores = self.get_nn_text(frame_texts, frame_feats, img_feats, 0)
        #print(sorted_frame_texts[0], " ", frame_scores[0])
        #if ppl_result == 'people':
        if sorted_frame_texts[0] == 'a high quality sharp image':
            frame_texts = ['sharp background', 'blurry background']
            frame_feats = self.get_text_feats([f'{p}.' for p in frame_texts])
            sorted_frame_texts, frame_scores = self.get_nn_text(frame_texts, frame_feats, img_feats, 0)
            if sorted_frame_texts[0] == 'blurry background':
                places = 0
            else:
                places = 1
            ppl_texts = ['no people', 'people']
            ppl_feats = self.get_text_feats([f'There are {p} in this photo.' for p in ppl_texts])
            sorted_ppl_texts, ppl_scores = self.get_nn_text(ppl_texts, ppl_feats, img_feats, 0)
            ppl_result = sorted_ppl_texts[0]
            if ppl_result == 'no people':
                people = 0
            else:
                people = 1
            ppl_texts = ['a lot of objects', 'no sharp objects']
            ppl_feats = self.get_text_feats([f'There are {p} in this photo.' for p in ppl_texts])
            sorted_ppl_texts, ppl_scores = self.get_nn_text(ppl_texts, ppl_feats, img_feats, 0)
            ppl_result = sorted_ppl_texts[0]
            if ppl_result == 'no sharp objects':
                objects = 0
            else:
                objects = 1
            return(1, places, people, objects)
        return(0, 0, 0, 0)

    def clip_persons_expert(self, frame, topk):
        img_feats = self.get_img_feats(frame)
        # Zero-shot VLM: classify number of people.
        ppl_texts = ['is one person', 'are two people', 'are three people', 'are several people', 'are many people']
        ppl_feats = self.get_text_feats([f'There {p} in this photo.' for p in ppl_texts])
        sorted_ppl_texts, ppl_scores = self.get_nn_text(ppl_texts, ppl_feats, img_feats, 0)
        ppl_result = sorted_ppl_texts[0]
        sorted_ppl_texts, ppl_scores = self.get_nn_text(self.people_texts, self.people_feats, img_feats, 0.11)
        return(sorted_ppl_texts[:topk], [ppl_result])

    def clip_location_expert(self, frame, topk):
        img_feats = self.get_img_feats(frame)
        sorted_places, places_scores = self.get_nn_text(self.place_texts, \
                                                        self.place_feats, img_feats, 0.14)
        return(sorted_places[:topk])
    
    def clip_cifar_expert(self, frame, topk):
        img_feats = self.get_img_feats(frame)
        sorted_cifar, cifar_scores = self.get_nn_text(self.cifar_texts, \
                                                        self.cifar_feats, img_feats, 0.15)
        return(sorted_cifar[:topk])

    def clip_objects_expert(self, frame, topk):
        img_feats = self.get_img_feats(frame)
        sorted_obj_texts, obj_scores = self.get_nn_text(self.object_texts, \
                                                        self.object_feats, img_feats, 0.18)
        return(sorted_obj_texts[:topk])

    def clip_experts_for_scene_element(self, movie_id):
        movie_info, fps, fn = self.clip_api.download_and_get_minfo(movie_id, to_print=True)
        if (fn):
            remote_api = self.clip_api.nre
            metadata = remote_api.get_movie_info(movie_id)
           
            video_file = Path(fn)
            file_name = fn
            print(file_name)
            first_frame = 0
            last_frame = metadata['scene_elements'][-1][1]
            if video_file.is_file():
                cap = cv2.VideoCapture(fn)
                locations_ = []
                scene_objects = {}
                scene_persons = {}
                scene_number_of_ppl = {}
                
                for scene_element, data in enumerate(metadata['scene_elements']):
                    print("Scene element: ", scene_element)
                    mdfs = metadata['mdfs'][scene_element]
                    objects_ = []
                    persons_ = []
                    number_of_ppl = []
                    for mdf in range(mdfs[0], mdfs[2]):
                    #for mdf in tqdm(range(first_frame, last_frame)):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, mdf)
                        ret, _frame_ = cap.read() # Read the frame
                        #for _frame__ in self.patch_frames_v(_frame_):
                            #for frame_ in self.patch_frames_h(_frame__):
                        scale_down_x = 0.25
                        scale_down_y = 0.25
                        frame_rgb = cv2.cvtColor(_frame_, cv2.COLOR_BGR2RGB)
                        #frame_res = cv2.resize(frame_rgb, None, fx= scale_down_x, fy= scale_down_y, interpolation= cv2.INTER_LINEAR)
                        if not ret:
                            print("File not found")
                        else:
                            #mdf_experts = self.clip_expert(frame_rgb, 3, 10)
                            good_frame, pcl, ppl, obj = self.mdf_selection(frame_rgb)
                            print("frame: ", good_frame, " place: ", pcl, " people: ", ppl, "objects: ", obj)
                            #frame_res = cv2.resize(frame_rgb, None, fx= scale_down_x, fy= scale_down_y, interpolation= cv2.INTER_LINEAR)
                            frame_res = frame_rgb
                            if good_frame  == 1:
                                if pcl == 1:
                                    for loc in self.clip_location_expert(frame_res, 10):
                                        locations_.append(loc)
                                if obj == 1:    
                                    for obj_ in self.clip_cifar_expert(frame_res, 10):
                                        objects_.append(obj_)
                                if ppl == 1:
                                    persons, number_of = self.clip_persons_expert(frame_res, 10)
                                    for pers in persons:
                                        persons_.append(pers)
                                    for nbr in number_of:
                                        number_of_ppl.append(nbr)
                            print("Frame # ", mdf)
                    counts = {item: objects_.count(item) for item in objects_}
                    sorted_objects = dict(sorted(counts.items(), key=lambda item: item[1], reverse = True))
                    sorted_objects = list(islice(sorted_objects, 10))
                    scene_objects[scene_element] = sorted_objects
                    counts = {item: persons_.count(item) for item in persons_}
                    sorted_persons = dict(sorted(counts.items(), key=lambda item: item[1], reverse = True))
                    sorted_persons = list(islice(sorted_persons, 10))
                    scene_persons[scene_element] = sorted_persons
                    
                counts = {item: locations_.count(item) for item in locations_}
                sorted_locations = dict(sorted(counts.items(), key=lambda item: item[1], reverse = True))
                return(list(islice(sorted_locations, 3)), scene_objects, scene_persons)


def main():
    cod = CLIP_OBJECT_DETECTOR()
    #clip.clip_encode_video('/home/dimas/0028_The_Crying_Game_00_53_53_876-00_53_55_522.mp4','Movies/114207205',0) Movies/114208196
    res = cod.clip_experts_for_scene_element('Movies/222511030')
    #res = cod.load_people_feats()
    print(res)


if __name__ == "__main__":
    main()