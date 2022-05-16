import clip
import cv2
import numpy as np
import torch
from PIL import Image
import subprocess
import os
import sys
from pathlib import Path
from vlm.clip_api import CLIP_API


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
        self.clip_api = clip_api
    
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

    def get_nn_text(self, raw_texts, text_feats, img_feats):
        scores = text_feats @ img_feats.T
        scores = scores.squeeze()
        high_to_low_ids = np.argsort(scores).squeeze()[::-1]
        high_to_low_texts = [raw_texts[i] for i in high_to_low_ids]
        high_to_low_scores = np.sort(scores).squeeze()[::-1]
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
        place_feats = self.get_text_feats([f'Photo of a {p}.' for p in place_texts])
        return(place_feats, place_texts)

    def load_object_feats(self, place_texts):
        if not os.path.exists('dictionary_and_semantic_hierarchy.txt'):
            subprocess.run(["/usr/bin/wget", "https://raw.githubusercontent.com/Tencent/tencent-ml-images/master/data/dictionary_and_semantic_hierarchy.txt"])
        with open('dictionary_and_semantic_hierarchy.txt') as fid:
            object_categories = fid.readlines()
        object_texts = []
#pf = ProfanityFilter()
        for object_text in object_categories[1:]:
            object_text = object_text.strip()
            object_text = object_text.split('\t')[3]
            safe_list = ''
            for variant in object_text.split(','):
                text = variant.strip()
                safe_list += f'{text}, '
            safe_list = safe_list[:-2]
            if len(safe_list) > 0:
                object_texts.append(safe_list)
        object_texts = [o for o in list(set(object_texts)) if o not in place_texts]  # Remove redundant categories.
        object_feats =self.get_text_feats([f'Photo of a {o}.' for o in object_texts])
        return(object_feats, object_texts)
    
    def clip_expert(self, frame, place_topk, obj_topk):
        img_feats = self.get_img_feats(frame)
        place_feats, place_texts = self.load_place_feats()
        object_feats, object_texts = self.load_object_feats(place_texts)
        img_types = ['photo', 'cartoon', 'sketch', 'painting', 'video', 'scene', 'shot', 'movie']
        img_types_feats = self.get_text_feats([f'This is a {t}.' for t in img_types])
        sorted_img_types, img_type_scores = self.get_nn_text(img_types, img_types_feats, img_feats)
        img_type = sorted_img_types[0]
        # Zero-shot VLM: classify number of people.
        ppl_texts = ['no people', 'people']
        ppl_feats = self.get_text_feats([f'There are {p} in this photo.' for p in ppl_texts])
        sorted_ppl_texts, ppl_scores = self.get_nn_text(ppl_texts, ppl_feats, img_feats)
        ppl_result = sorted_ppl_texts[0]
        if ppl_result == 'people':
            ppl_texts = ['is one person', 'are two people', 'are three people', 'are several people', 'are many people']
            ppl_feats = self.get_text_feats([f'There {p} in this photo.' for p in ppl_texts])
            sorted_ppl_texts, ppl_scores = self.get_nn_text(ppl_texts, ppl_feats, img_feats)
            ppl_result = sorted_ppl_texts[0]
        else:
            ppl_result = f'are {ppl_result}'
        # Zero-shot VLM: classify places.
        place_feats = self.get_text_feats([f'Scene of a {p}.' for p in place_texts ])
        sorted_places, places_scores = self.get_nn_text(place_texts, place_feats, img_feats)
        # Zero-shot VLM: classify objects.
        sorted_obj_texts, obj_scores = self.get_nn_text(object_texts, object_feats, img_feats)
        object_list = ''
        for i in range(obj_topk):
            object_list += f'{sorted_obj_texts[i]}, '
        object_list = object_list[:-2]
        return(sorted_places[:place_topk], object_list, ppl_result)
    
    def clip_experts_for_moive(self, movie_id, scene_element):
        movie_info, fps, fn = self.clip_api.download_and_get_minfo(movie_id, to_print=True)
        if (fn):
            remote_api = self.clip_api.nre
            metadata = remote_api.get_movie_info(movie_id)
            mdfs = metadata['mdfs'][scene_element]
            video_file = Path(fn)
            file_name = fn
            print(file_name)
            if video_file.is_file():
                cap = cv2.VideoCapture(fn)
                scene_experts = []
                for count, mdf in enumerate(mdfs):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, mdf)
                    ret, frame_ = cap.read() # Read the frame
                    frame_rgb = cv2.cvtColor(frame_, cv2.COLOR_BGR2RGB)
                    if not ret:
                        print("File not found")
                    else:
                        mdf_experts = self.clip_expert(frame_rgb, 5, 20)
                        scene_experts.append(mdf_experts)
                return(scene_experts)

def main():
    cod=CLIP_OBJECT_DETECTOR()
    #clip.clip_encode_video('/home/dimas/0028_The_Crying_Game_00_53_53_876-00_53_55_522.mp4','Movies/114207205',0)
    res = cod.clip_experts_for_moive('Movies/114207205', 0)
    print(res)
if __name__ == "__main__":
    main()