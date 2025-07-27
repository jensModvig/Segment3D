import torch
from torch_scatter import scatter_mean

class TrainerUtilsMixin:
    def get_mask_and_scores(self, mask_cls, mask_pred, num_queries=100, num_classes=18, device=None):
        if device is None:
            device = self.device
            
        # Check for filter-first strategy (Class 3)
        if hasattr(self, '_filter_first') and self._filter_first:
            return self._filter_first_strategy(mask_cls, mask_pred, num_queries, num_classes, device)
        else:
            # Use topk-first strategy (Class 1 and 2)
            use_valid_filter = hasattr(self, '_use_valid_filter') and self._use_valid_filter
            return self._topk_first_strategy(mask_cls, mask_pred, num_queries, num_classes, device, use_valid_filter)
    
    def _filter_first_strategy(self, mask_cls, mask_pred, num_queries, num_classes, device):
        """Implements Class 3 logic: filter valid columns first, then topk"""
        labels = torch.arange(num_classes, device=device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
        
        result_pred_mask = (mask_pred > 0).float()
        # Filter out columns where no predictions were made
        valid_cols = result_pred_mask.sum(0) > 0
        
        if valid_cols.sum() == 0:
            return self._empty_results(mask_pred, device)
            
        labels = labels[valid_cols]
        mask_pred = mask_pred[:, valid_cols]
        mask_cls = mask_cls[valid_cols]
        result_pred_mask = result_pred_mask[:, valid_cols]
        
        heatmap = mask_pred.float().sigmoid()
        mask_scores_per_image = (heatmap * result_pred_mask).sum(0) / (result_pred_mask.sum(0) + 1e-6)
        score = mask_cls.flatten(0, 1) * mask_scores_per_image
        
        if self.config.general.topk_per_image == -1:
            topk = num_queries
        else:
            topk = int(self.config.general.topk_per_image * num_queries)
        
        # Ensure topk doesn't exceed available elements after filtering
        topk = min(topk, score.size(0))
        
        if topk <= 0:
            # Handle case with no valid predictions
            return self._empty_results(mask_pred, device)
            
        score, topk_indices = score.topk(topk, sorted=True)
        classes = labels[topk_indices]
        result_pred_mask = result_pred_mask[:, topk_indices]
        heatmap = heatmap[:, topk_indices]
        
        return score, result_pred_mask, classes, heatmap
    
    def _empty_results(self, mask_pred, device):
        """Return empty results when no valid predictions are found"""
        return (
            torch.tensor([], device=device),
            torch.zeros((mask_pred.size(0), 0), device=device),
            torch.tensor([], dtype=torch.long, device=device),
            torch.zeros((mask_pred.size(0), 0), device=device)
        )
    
    def _topk_first_strategy(self, mask_cls, mask_pred, num_queries, num_classes, device, use_valid_filter):
        """Implements Class 1 and 2 logic: topk first, then optionally filter valid columns"""
        labels = torch.arange(num_classes, device=device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
        
        available = mask_cls.flatten(0, 1).size(0)
        if self.config.general.topk_per_image != -1:
            topk = min(self.config.general.topk_per_image, available)
        else:
            topk = min(num_queries, available)
            
        if topk <= 0:
            return self._empty_results(mask_pred, device)
            
        scores_per_query, topk_indices = mask_cls.flatten(0, 1).topk(topk, sorted=True)
            
        labels_per_query = labels[topk_indices]
        topk_indices = topk_indices // num_classes
        mask_pred = mask_pred[:, topk_indices]
        result_pred_mask = (mask_pred > 0).float()
        
        # Class 1 has valid column filtering, Class 2 doesn't
        if use_valid_filter:
            valid_cols = result_pred_mask.sum(0) > 0
            if valid_cols.sum() == 0:
                return self._empty_results(mask_pred, device)
            labels_per_query = labels_per_query[valid_cols]
            mask_pred = mask_pred[:, valid_cols]
            scores_per_query = scores_per_query[valid_cols]
            result_pred_mask = result_pred_mask[:, valid_cols]
        
        heatmap = mask_pred.float().sigmoid()
        mask_scores_per_image = (heatmap * result_pred_mask).sum(0) / (result_pred_mask.sum(0) + 1e-6)
        score = scores_per_query * mask_scores_per_image
        classes = labels_per_query
        
        return score, result_pred_mask, classes, heatmap

    def get_full_res_mask(self, mask, inverse_map, point2segment_full, is_heatmap=False):
        mask = mask.detach().cpu()[inverse_map]
        if self.eval_on_segments and not is_heatmap:
            mask = scatter_mean(mask, point2segment_full, dim=0)
            mask = (mask > 0.5).float()
            mask = mask.detach().cpu()[point2segment_full.cpu()]
        return mask