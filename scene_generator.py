import os
import math
import random
from collections import Counter
from glob import glob
from pathlib import Path

import numpy as np
import matplotlib.image as mpimg
import pickle


import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.kps import KeypointsOnImage
from imgaug.augmentables.polys import Polygon
from imgaug.augmentables.segmaps import SegmentationMapOnImage

BACKGROUNDS_PICKLE = 'data/backgrounds.pck'
BACKGROUNDS_FOLDER = 'data/dtd'
CARDS_PICKLE = 'data/cards.pck'
LABELS_PATH = 'data/labels.txt'
imgH = imgW = 704

class Backgrounds():
    def __init__(self, backgrounds_pck=BACKGROUNDS_PICKLE, backgrounds_folder=None):
        if backgrounds_folder:
            self._images = []
            for f in glob(backgrounds_folder + '/images/banded/*.jpg'):
                    self._images.append(mpimg.imread(f))
        else:
            self._images = pickle.load(open(backgrounds_pck,'rb'))
        self._nb_images=len(self._images)
        print("Nb of backgrounds loaded :", self._nb_images)
    def get_random(self, display=False):
        bg = self._images[random.randint(0,self._nb_images-1)]
        return bg


class Cards():
    def __init__(self, cards_pck=CARDS_PICKLE):
        self._cards=pickle.load(open(cards_pck,'rb'))
        # self._cards is a dictionary where keys are card names (ex:'Kc') and values are lists of (img,hullHL,hullLR) 
        self._nb_cards_by_value={k:len(self._cards[k]) for k in self._cards}
        # print("Nb of cards loaded per name :", self._nb_cards_by_value)
        
    def get_random(self, card_name=None, display=False):
        if card_name is None:
            card_name= random.choice(list(self._cards.keys()))
        card, hull =self._cards[card_name][random.randint(0,self._nb_cards_by_value[card_name]-1)]
        return card, card_name, hull


class Scenes():
    def __init__(self, cards, backgrounds, labels=None):
        self.imgH = 704
        self.imgW = 704
        self.ratio = 0.8
        self.labels = labels
        self.cards = cards
        self.backgrounds = backgrounds
        self.resize_bg = iaa.Resize({"height": self.imgH, "width": self.imgW})
        self.seq = iaa.Sequential([
            iaa.Affine(scale=[0.65,1]),
            iaa.Affine(rotate=(-180,180)),
            iaa.Affine(translate_percent={"x":(-0.25,0.25),"y":(-0.25,0.25)}),
        ])


    def augment_card(self, card, hull):
        cardH, cardW, _ = card.shape
        decalX=int((imgW-cardW)/2)
        decalY=int((imgH-cardH)/2)

        img_card = np.zeros((imgH, imgW, 4), dtype=np.uint8)
        img_card[decalY:decalY+cardH,decalX:decalX+cardW,:] = card

        card = img_card[:,:,:3]
        hull = hull[:,0,:]

        card_kps_xy = np.float32([
            [decalX, decalY],
            [decalX + cardW, decalY],
            [decalX + cardW, decalY + cardH],
            [decalX, decalY + cardH]
        ])
        kpsoi_card = KeypointsOnImage.from_xy_array(card_kps_xy, shape=card.shape)

        # hull is a cv2.Contour, shape : Nx1x2
        kpsoi_hull = [ia.Keypoint(x=p[0]+decalX, y=p[1]+decalY) for p in hull.reshape(-1,2)]
        kpsoi_hull = KeypointsOnImage(kpsoi_hull, shape=(imgH,imgW,3))

        # create polygon
        poly_hull = Polygon(kpsoi_hull.keypoints)

        # create empty segmentation map for classes: background, tree, chipmunk
        segmap = np.zeros((card.shape[0], card.shape[1], 3), dtype=np.uint8)

        # draw the tree polygon into the second channel
        segmap = poly_hull.draw_on_image(
            segmap,
            color=(0, 255, 0),
            alpha=1.0, alpha_lines=0.0, alpha_points=0.0)

        # merge the two channels to a single one
        segmap = np.argmax(segmap, axis=2)
        segmap = SegmentationMapOnImage(segmap, nb_classes=2, shape=card.shape)

        myseq = self.seq.to_deterministic()
        card_aug, segmap_aug = myseq(image=card, segmentation_maps=segmap)
        card_aug, kpsoi_aug = myseq(image=card, keypoints=kpsoi_card)

        return card_aug, kpsoi_aug, segmap_aug

    
    def kps_to_mask(self, kpsoi):
        poly_card = Polygon(kpsoi.keypoints)

        segmap = np.zeros((imgH, imgW, 3), dtype=np.uint8)

        # draw the tree polygon into the second channel
        segmap = poly_card.draw_on_image(
            segmap,
            color=(0, 255, 0),
            alpha=1.0, alpha_lines=0.0, alpha_points=0.0)

        # merge the two channels to a single one
        segmap = np.argmax(segmap, axis=2)
        card_mask = np.stack([segmap]*3,-1)

        return card_mask


    def get_random(self):
        bg = self.backgrounds.get_random()
        bg = self.resize_bg.augment_image(bg)

        while True:
            card1, card_name1, hull1 = self.cards.get_random()
            card_aug1, kpsoi_aug1, segmap_aug1 = self.augment_card(card1, hull1)
            card_mask1 = self.kps_to_mask(kpsoi_aug1)

            card2, card_name2, hull2 = self.cards.get_random()
            card_aug2, kpsoi_aug2, segmap_aug2 = self.augment_card(card2, hull2)
            card_mask2 = self.kps_to_mask(kpsoi_aug2)

            # Handle superposition
            arr = segmap_aug1.get_arr_int()
            original_size = np.sum(arr)
            arr = np.where(card_mask2[:, :, 0], 0, arr)
            segmap_aug1 = SegmentationMapOnImage(arr, nb_classes=2, shape=bg.shape)
            new_size = np.sum(arr)

            final = np.where(card_mask1, card_aug1, bg)
            final = np.where(card_mask2, card_aug2, final)
            if new_size < original_size * self.ratio:
                continue
            break

        # create empty segmentation map for classes: background, tree, chipmunk
        mask_final = np.zeros((imgH, imgW, 2), dtype=np.uint8)
        mask_final[:, :, 0] = segmap_aug1.get_arr_int()
        mask_final[:, :, 1] = segmap_aug2.get_arr_int()
        
        # Add classes
        class_id = np.array([self.labels.index(card_name1) + 1,
                             self.labels.index(card_name2) + 1])

        return final, (mask_final, class_id)

with open(LABELS_PATH) as f:
    labels = f.read().splitlines()
backgrounds = Backgrounds(backgrounds_folder=BACKGROUNDS_FOLDER)
cards = Cards()
scene = Scenes(cards, backgrounds, labels=labels)

scenes = []
classes = []
for _ in range(5):
    image_and_masks = scene.get_random()
    classes.extend(list(image_and_masks[1][1]))

class_names = list(map(lambda x: labels[x-1], classes))
print(class_names)
print(Counter(class_names))