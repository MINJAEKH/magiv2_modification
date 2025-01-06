from transformers import PreTrainedModel, VisionEncoderDecoderModel, ViTMAEModel, ConditionalDetrModel, ConditionalDetrConfig ##add 
from transformers.models.conditional_detr.modeling_conditional_detr import (
    ConditionalDetrMLPPredictionHead, 
    ConditionalDetrModelOutput,
    ## delete 
    inverse_sigmoid,
)
from configuration_magiv2 import Magiv2Config
from processing_magiv2 import Magiv2Processor
from torch import nn
from typing import Optional, List
import torch
from einops import rearrange, repeat
from utils import move_to_device, visualise_single_image_prediction, sort_panels
from transformers.image_transforms import center_to_corners_format
from utils import UnionFind, sort_panels
import pulp
import scipy
import numpy as np

class Magiv2Model(PreTrainedModel):
    config_class = Magiv2Config

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.processor = Magiv2Processor(config)
        if not config.disable_ocr:
            self.ocr_model = VisionEncoderDecoderModel(config.ocr_model_config)
        if not config.disable_crop_embeddings:
            self.crop_embedding_model = ViTMAEModel(config.crop_embedding_model_config)
        if not config.disable_detections:
            self.num_non_obj_tokens = 5
            self.detection_transformer = ConditionalDetrModel(config.detection_model_config)
            self.bbox_predictor = ConditionalDetrMLPPredictionHead(
                input_dim=config.detection_model_config.d_model,
                hidden_dim=config.detection_model_config.d_model,
                output_dim=4, num_layers=3
            )
            self.character_character_matching_head = ConditionalDetrMLPPredictionHead(
                input_dim = 3 * config.detection_model_config.d_model + (2 * config.crop_embedding_model_config.hidden_size if not config.disable_crop_embeddings else 0),
                hidden_dim=config.detection_model_config.d_model,
                output_dim=1, num_layers=3
            )
            
            self.class_labels_classifier = nn.Linear(
                config.detection_model_config.d_model, config.detection_model_config.num_labels
            )
        
            self.matcher = ConditionalDetrModel(ConditionalDetrConfig( ##modify
                class_cost=config.detection_model_config.class_cost,
                bbox_cost=config.detection_model_config.bbox_cost,
                giou_cost=config.detection_model_config.giou_cost
            ))

    def move_to_device(self, input):
        return move_to_device(input, self.device)
    
    @torch.no_grad()
    def do_chapter_wide_prediction(self, pages_in_order, character_bank, eta=0.75, batch_size=8, use_tqdm=False, do_ocr=True):
        texts = []
        characters = []
        character_clusters = []
        if use_tqdm:
            from tqdm import tqdm
            iterator = tqdm(range(0, len(pages_in_order), batch_size))
        else:
            iterator = range(0, len(pages_in_order), batch_size)
        per_page_results = []
        for i in iterator:
            pages = pages_in_order[i:i+batch_size]
            results = self.predict_detections_and_associations(pages)
            per_page_results.extend([result for result in results])

        characters = [result["characters"] for result in per_page_results]
        character_clusters = [result["character_cluster_labels"] for result in per_page_results]
        assigned_character_names = self.assign_names_to_characters(pages_in_order, characters, character_bank, character_clusters, eta=eta)
        
        offset_characters = 0
        for result in per_page_results :
            result['character_names'] = assigned_character_names[offset_characters:offset_characters + len(result["characters"])]
            offset_characters += len(result['characters'])
            
        return per_page_results
        
    
    def assign_names_to_characters(self, images, character_bboxes, character_bank, character_clusters, eta=0.75):
        if len(character_bank["images"]) == 0:
            return ["Other" for bboxes_for_image in character_bboxes for bbox in bboxes_for_image]
        chapter_wide_char_embeddings = self.predict_crop_embeddings(images, character_bboxes)
        chapter_wide_char_embeddings = torch.cat(chapter_wide_char_embeddings, dim=0)
        chapter_wide_char_embeddings = torch.nn.functional.normalize(chapter_wide_char_embeddings, p=2, dim=1).cpu().numpy()
        # create must-link and cannot link constraints from character_clusters
        must_link = []
        cannot_link = []
        offset = 0
        for clusters_per_image in character_clusters:
            for i in range(len(clusters_per_image)):
                for j in range(i+1, len(clusters_per_image)):
                    if clusters_per_image[i] == clusters_per_image[j]:
                        must_link.append((offset + i, offset + j))
                    else:
                        cannot_link.append((offset + i, offset + j))
            offset += len(clusters_per_image)
        character_bank_for_this_chapter = self.predict_crop_embeddings(character_bank["images"], [[[0, 0, x.shape[1], x.shape[0]]] for x in character_bank["images"]])
        character_bank_for_this_chapter = torch.cat(character_bank_for_this_chapter, dim=0)
        character_bank_for_this_chapter = torch.nn.functional.normalize(character_bank_for_this_chapter, p=2, dim=1).cpu().numpy()
        costs = scipy.spatial.distance.cdist(chapter_wide_char_embeddings, character_bank_for_this_chapter)
        none_of_the_above = eta * np.ones((costs.shape[0],1))
        costs = np.concatenate([costs, none_of_the_above], axis=1)
        sense = pulp.LpMinimize
        num_supply, num_demand = costs.shape
        problem = pulp.LpProblem("Optimal_Transport_Problem", sense)
        x = pulp.LpVariable.dicts("x", ((i, j) for i in range(num_supply) for j in range(num_demand)), cat='Binary')
        # Objective Function to minimize
        problem += pulp.lpSum([costs[i][j] * x[(i, j)] for i in range(num_supply) for j in range(num_demand)])
        # each crop must be assigned to exactly one character
        for i in range(num_supply):
            problem += pulp.lpSum([x[(i, j)] for j in range(num_demand)]) == 1, f"Supply_{i}_Total_Assignment"
        # cannot link constraints
        for j in range(num_demand-1):
            for (s1, s2) in cannot_link:
                problem += x[(s1, j)] + x[(s2, j)] <= 1, f"Exclusion_{s1}_{s2}_Demand_{j}"
        # must link constraints
        for j in range(num_demand):
            for (s1, s2) in must_link:
                problem += x[(s1, j)] - x[(s2, j)] == 0, f"Inclusion_{s1}_{s2}_Demand_{j}"
        problem.solve()
        assignments = []
        for v in problem.variables():
            if v.varValue > 0:
                index, assignment = v.name.split("(")[1].split(")")[0].split(",")
                assignment = assignment[1:]
                assignments.append((int(index), int(assignment)))

        labels = np.zeros(num_supply)
        for i, j in assignments:
            labels[i] = j
        
        return [character_bank["names"][int(i)] if i < len(character_bank["names"]) else "Other" for i in labels]

    
    def predict_detections_and_associations(
            self,
            images,
            move_to_device_fn=None,
            character_detection_threshold=0.3,
            panel_detection_threshold=0.2,
            character_character_matching_threshold=0.65,
        ):
        assert not self.config.disable_detections
        move_to_device_fn = self.move_to_device if move_to_device_fn is None else move_to_device_fn
        
        inputs_to_detection_transformer = self.processor.preprocess_inputs_for_detection(images)
        inputs_to_detection_transformer = move_to_device_fn(inputs_to_detection_transformer)
        
        detection_transformer_output = self._get_detection_transformer_output(**inputs_to_detection_transformer)
        predicted_class_scores, predicted_bboxes = self._get_predicted_bboxes_and_classes(detection_transformer_output)

        original_image_sizes = torch.stack([torch.tensor(img.shape[:2]) for img in images], dim=0).to(predicted_bboxes.device)

        batch_scores, batch_labels = predicted_class_scores.max(-1)
        batch_scores = batch_scores.sigmoid()
        batch_labels = batch_labels.long()
        batch_bboxes = center_to_corners_format(predicted_bboxes)

        # scale the bboxes back to the original image size
        if isinstance(original_image_sizes, List):
            img_h = torch.Tensor([i[0] for i in original_image_sizes])
            img_w = torch.Tensor([i[1] for i in original_image_sizes])
        else:
            img_h, img_w = original_image_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(batch_bboxes.device)
        batch_bboxes = batch_bboxes * scale_fct[:, None, :]
        
        batch_panel_indices = self.processor._get_indices_of_panels_to_keep(batch_scores, batch_labels, batch_bboxes, panel_detection_threshold)
        batch_character_indices = self.processor._get_indices_of_characters_to_keep(batch_scores, batch_labels, batch_bboxes, character_detection_threshold)

        predicted_obj_tokens_for_batch = self._get_predicted_obj_tokens(detection_transformer_output)
        predicted_c2c_tokens_for_batch = self._get_predicted_c2c_tokens(detection_transformer_output)

        character_bboxes_in_batch = [batch_bboxes[i][j] for i, j in enumerate(batch_character_indices)]
        character_character_affinity_matrices = self._get_character_character_affinity_matrices(
            character_obj_tokens_for_batch=[x[i] for x, i in zip(predicted_obj_tokens_for_batch, batch_character_indices)],
            crop_embeddings_for_batch=self.predict_crop_embeddings(images, character_bboxes_in_batch, move_to_device_fn),
            c2c_tokens_for_batch=predicted_c2c_tokens_for_batch,
            apply_sigmoid=True,
        )

        results = []
        for batch_index in range(len(batch_scores)):
            panel_indices = batch_panel_indices[batch_index]
            character_indices = batch_character_indices[batch_index]

            character_bboxes = batch_bboxes[batch_index][character_indices]
            panel_bboxes = batch_bboxes[batch_index][panel_indices]

            local_sorted_panel_indices = sort_panels(panel_bboxes)
            panel_bboxes = panel_bboxes[local_sorted_panel_indices]

            character_character_matching_scores = character_character_affinity_matrices[batch_index]

            character_cluster_labels = UnionFind.from_adj_matrix(
                character_character_matching_scores > character_character_matching_threshold
            ).get_labels_for_connected_components()

            results.append({
                "panels": panel_bboxes.tolist(),
                "characters": character_bboxes.tolist(),
                "character_cluster_labels": character_cluster_labels,
            })

        return results

    def get_affinity_matrices_given_annotations(
            self, images, annotations, move_to_device_fn=None, apply_sigmoid=True
    ):
        assert not self.config.disable_detections
        move_to_device_fn = self.move_to_device if move_to_device_fn is None else move_to_device_fn

        character_bboxes_in_batch = [[bbox for bbox, label in zip(a["bboxes_as_x1y1x2y2"], a["labels"]) if label == 0] for a in annotations]
        crop_embeddings_for_batch = self.predict_crop_embeddings(images, character_bboxes_in_batch, move_to_device_fn)

        inputs_to_detection_transformer = self.processor.preprocess_inputs_for_detection(images, annotations)
        inputs_to_detection_transformer = move_to_device_fn(inputs_to_detection_transformer)
        processed_targets = inputs_to_detection_transformer.pop("labels")

        detection_transformer_output = self._get_detection_transformer_output(**inputs_to_detection_transformer)
        predicted_obj_tokens_for_batch = self._get_predicted_obj_tokens(detection_transformer_output)
        predicted_c2c_tokens_for_batch = self._get_predicted_c2c_tokens(detection_transformer_output)

        predicted_class_scores, predicted_bboxes = self._get_predicted_bboxes_and_classes(detection_transformer_output)
        matching_dict = {
            "logits": predicted_class_scores,
            "pred_boxes": predicted_bboxes,
        }
        indices = self.matcher(matching_dict, processed_targets)

        matched_char_obj_tokens_for_batch = []
        c2c_tokens_for_batch = []

        for j, (pred_idx, tgt_idx) in enumerate(indices):
            target_idx_to_pred_idx = {tgt.item(): pred.item() for pred, tgt in zip(pred_idx, tgt_idx)}
            targets_for_this_image = processed_targets[j]
            indices_of_char_boxes_in_annotation = [i for i, label in enumerate(targets_for_this_image["class_labels"]) if label == 0]
            predicted_char_indices = [target_idx_to_pred_idx[i] for i in indices_of_char_boxes_in_annotation]
            matched_char_obj_tokens_for_batch.append(predicted_obj_tokens_for_batch[j][predicted_char_indices])
            c2c_tokens_for_batch.append(predicted_c2c_tokens_for_batch[j])

        character_character_affinity_matrices = self._get_character_character_affinity_matrices(
            character_obj_tokens_for_batch=matched_char_obj_tokens_for_batch,
            crop_embeddings_for_batch=crop_embeddings_for_batch,
            c2c_tokens_for_batch=c2c_tokens_for_batch,
            apply_sigmoid=apply_sigmoid,
        )
        
        character_character_affinity_matrices_crop_only = self._get_character_character_affinity_matrices(
            character_obj_tokens_for_batch=matched_char_obj_tokens_for_batch,
            crop_embeddings_for_batch=crop_embeddings_for_batch,
            c2c_tokens_for_batch=c2c_tokens_for_batch,
            crop_only=True,
            apply_sigmoid=apply_sigmoid,
        )

        return {
            "character_character_affinity_matrices": character_character_affinity_matrices,
            "character_character_affinity_matrices_crop_only": character_character_affinity_matrices_crop_only,
        }

    
    def predict_crop_embeddings(self, images, crop_bboxes, move_to_device_fn=None, mask_ratio=0.0, batch_size=256):
        if self.config.disable_crop_embeddings:
            return None
        
        assert isinstance(crop_bboxes, List), "please provide a list of bboxes for each image to get embeddings for"
        
        move_to_device_fn = self.move_to_device if move_to_device_fn is None else move_to_device_fn
        
        # temporarily change the mask ratio from default to the one specified
        old_mask_ratio = self.crop_embedding_model.embeddings.config.mask_ratio
        self.crop_embedding_model.embeddings.config.mask_ratio = mask_ratio

        crops_per_image = []
        num_crops_per_batch = [len(bboxes) for bboxes in crop_bboxes]
        for image, bboxes, num_crops in zip(images, crop_bboxes, num_crops_per_batch):
            crops = self.processor.crop_image(image, bboxes)
            assert len(crops) == num_crops
            crops_per_image.extend(crops)
        
        if len(crops_per_image) == 0:
            return [move_to_device_fn(torch.zeros(0, self.config.crop_embedding_model_config.hidden_size)) for _ in crop_bboxes]

        crops_per_image = self.processor.preprocess_inputs_for_crop_embeddings(crops_per_image)
        crops_per_image = move_to_device_fn(crops_per_image)
        
        # process the crops in batches to avoid OOM
        embeddings = []
        for i in range(0, len(crops_per_image), batch_size):
            crops = crops_per_image[i:i+batch_size]
            embeddings_per_batch = self.crop_embedding_model(crops).last_hidden_state[:, 0]
            embeddings.append(embeddings_per_batch)
        embeddings = torch.cat(embeddings, dim=0)

        crop_embeddings_for_batch = []
        for num_crops in num_crops_per_batch:
            crop_embeddings_for_batch.append(embeddings[:num_crops])
            embeddings = embeddings[num_crops:]
        
        # restore the mask ratio to the default
        self.crop_embedding_model.embeddings.config.mask_ratio = old_mask_ratio

        return crop_embeddings_for_batch
    
    def visualise_single_image_prediction(
            self, image_as_np_array, predictions, filename=None
    ):
        return visualise_single_image_prediction(image_as_np_array, predictions, filename)

    
    @torch.no_grad()
    def _get_detection_transformer_output(
            self, 
            pixel_values: torch.FloatTensor,
            pixel_mask: Optional[torch.LongTensor] = None
    ):
        if self.config.disable_detections:
            raise ValueError("Detection model is disabled. Set disable_detections=False in the config.")
        return self.detection_transformer(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            return_dict=True
        )
    
    def _get_predicted_obj_tokens(
            self,
            detection_transformer_output: ConditionalDetrModelOutput
    ):
        return detection_transformer_output.last_hidden_state[:, :-self.num_non_obj_tokens]
    
    def _get_predicted_c2c_tokens(
            self,
            detection_transformer_output: ConditionalDetrModelOutput
    ):
        return detection_transformer_output.last_hidden_state[:, -self.num_non_obj_tokens]
    
    def _get_predicted_bboxes_and_classes(
            self,
            detection_transformer_output: ConditionalDetrModelOutput,
    ):
        if self.config.disable_detections:
            raise ValueError("Detection model is disabled. Set disable_detections=False in the config.")

        obj = self._get_predicted_obj_tokens(detection_transformer_output)

        predicted_class_scores = self.class_labels_classifier(obj)
        reference = detection_transformer_output.reference_points[:-self.num_non_obj_tokens] 
        reference_before_sigmoid = inverse_sigmoid(reference).transpose(0, 1)
        predicted_boxes = self.bbox_predictor(obj)
        predicted_boxes[..., :2] += reference_before_sigmoid
        predicted_boxes = predicted_boxes.sigmoid()

        return predicted_class_scores, predicted_boxes
    
    def _get_character_character_affinity_matrices(
            self,
            character_obj_tokens_for_batch: List[torch.FloatTensor] = None,
            crop_embeddings_for_batch: List[torch.FloatTensor] = None,
            c2c_tokens_for_batch: List[torch.FloatTensor] = None,
            crop_only=False,
            apply_sigmoid=True,
    ):
        assert self.config.disable_detections or (character_obj_tokens_for_batch is not None and c2c_tokens_for_batch is not None)
        assert self.config.disable_crop_embeddings or crop_embeddings_for_batch is not None
        assert not self.config.disable_detections or not self.config.disable_crop_embeddings

        if crop_only:
            affinity_matrices = []
            for crop_embeddings in crop_embeddings_for_batch:
                crop_embeddings = crop_embeddings / crop_embeddings.norm(dim=-1, keepdim=True)
                affinity_matrix = crop_embeddings @ crop_embeddings.T
                affinity_matrices.append(affinity_matrix)
            return affinity_matrices
        affinity_matrices = []
        for batch_index, (character_obj_tokens, c2c) in enumerate(zip(character_obj_tokens_for_batch, c2c_tokens_for_batch)):
            if character_obj_tokens.shape[0] == 0:
                affinity_matrices.append(torch.zeros(0, 0).type_as(character_obj_tokens))
                continue
            if not self.config.disable_crop_embeddings:
                crop_embeddings = crop_embeddings_for_batch[batch_index]
                assert character_obj_tokens.shape[0] == crop_embeddings.shape[0]
                character_obj_tokens = torch.cat([character_obj_tokens, crop_embeddings], dim=-1)
            char_i = repeat(character_obj_tokens, "i d -> i repeat d", repeat=character_obj_tokens.shape[0])
            char_j = repeat(character_obj_tokens, "j d -> repeat j d", repeat=character_obj_tokens.shape[0])
            char_ij = rearrange([char_i, char_j], "two i j d -> (i j) (two d)")
            c2c = repeat(c2c, "d -> repeat d", repeat = char_ij.shape[0])
            char_ij_c2c = torch.cat([char_ij, c2c], dim=-1)
            character_character_affinities = self.character_character_matching_head(char_ij_c2c)
            character_character_affinities = rearrange(character_character_affinities, "(i j) 1 -> i j", i=char_i.shape[0])
            character_character_affinities = (character_character_affinities + character_character_affinities.T) / 2
            if apply_sigmoid:
                character_character_affinities = character_character_affinities.sigmoid()
            affinity_matrices.append(character_character_affinities)
        return affinity_matrices
