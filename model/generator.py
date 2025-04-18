import numpy as np
from model.utils import mean_embeddings, similarity_score


def nomalize(arr):
    min_val = min(arr)
    max_val = max(arr)

    normalized = [(x - min_val) / (max_val - min_val) for x in arr]

    return normalized

class Summarizer:
    def __init__(self, scoring_mode, kf_mode, k):
        print(f"Summarizer's scoring mode is {scoring_mode}")
        print(f"Input KF mode is {kf_mode}")
        
        self.scoring_mode = scoring_mode
        self.k = k
        
        self.kf_mode = []
        if scoring_mode == 'mean':
            if 'mean' in kf_mode:
                self.kf_mode.append('mean')
        
        if 'middle' in kf_mode:
            self.kf_mode.append('middle')
        
        if 'ends' in kf_mode:
            self.kf_mode.append('ends')
            
        print(f"Summarizer's KF mode is {self.kf_mode}")

    # For each segment, compute the mean features and
    # similarity of all features with the mean
    def score_segments(self, embeddings, segments, bias):
        segment_scores = []
        
        for _, start, end in segments:
            # Get the associated features
            segment_features = embeddings[start:end]
            
            # Calculate the scores for frames in the segment
            if self.scoring_mode == "uniform":
                # Give bias to frames closer to the keyframes in positions
                filling = len(segment_features)
                max_score = filling * (1 + bias)
                min_score = filling
                
                if bias < 0:
                    max_score, min_score = min_score, max_score
                
                # Scores of frames are a cosine curve between nearest keyframes
                period = None
                if "middle" in self.kf_mode and "ends" in self.kf_mode:
                    period = 4 * np.pi
                elif "middle" in self.kf_mode or "ends" in self.kf_mode:
                    period = 2 * np.pi
                
                if period is not None:
                    start_phase = 0 if 'ends' in self.kf_mode else np.pi
                    end_phase = start_phase + period
                    magnitude = (max_score - min_score) / 2
                    domain = np.linspace(start_phase, end_phase,
                                         end - start)
                    
                    position_score = magnitude * np.cos(domain) + magnitude + min_score
                else:
                    position_score = [min_score] * len(segment_features)
                
                representative = mean_embeddings(segment_features)
                
                content_score = similarity_score(segment_features,
                                         representative).tolist()
                
                nomalized_content_score = nomalize(content_score)
                step_value = max(position_score) - min(position_score)

                
                score = np.asarray(position_score) +  self.k * np.asarray(nomalized_content_score) * step_value
            
            else:
                if self.scoring_mode == "mean":
                    representative = mean_embeddings(segment_features)
                else:
                    representative = segment_features[len(segment_features)
                                                      // 2]
                
                score = similarity_score(segment_features,
                                         representative).tolist()
            
            segment_scores.extend(score)
        
        return np.asarray(segment_scores)

    def select_keyframes(self, segments, scores, length):
        keyframe_indices = []
        
        for _, start, end in segments:
            if 'mean' in self.kf_mode:
                segment_scores = scores[start:end]
                keyframe_indices.append(np.argmax(segment_scores) + start)
            
            if 'middle' in self.kf_mode:
                keyframe_indices.append((start + end) // 2)
                
            if 'ends' in self.kf_mode:
                keyframe_indices.append(start)
                keyframe_indices.append(end - 1)

        if length > 0:
            unselected_indicies = np.setdiff1d(np.arange(len(scores)),
                                               keyframe_indices)
            
            unselected_scores = scores[unselected_indicies]
            
            remained_length = length - len(keyframe_indices)
            unselected_keyframes = np.argpartition(unselected_scores,
                                                   -remained_length)[-remained_length:]
            
            keyframe_indices.extend(unselected_indicies[unselected_keyframes])
        
        return np.sort(keyframe_indices)
    
