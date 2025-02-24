import torch.nn as nn 
from transformers import BertModel
import torchvision.models.video as vision_models
import torch
from meld_dataset import MELDDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

class TextModel(nn.Module):
    def __init__(self):
        super(TextModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        for param in self.bert.parameters():
            param.requires_grad = False
        
        self.projection = nn.Linear(768, 128)

    
    def forward(self, input_ids, attention_mask):
        # Extract bert embeddings
        outputs = self.bert(input_ids, attention_mask)

        # Use the [CLS] token embedding
        pooled_output = outputs.pooler_output

        return self.projection(pooled_output)

class VideoEncoder(nn.Module):
    def __init__(self):
        self.backbone = vision_models.r3d_18(pretrained=True)

        for param in self.backbone.parameters():
            param.requires_grad = False

        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        # [batch_size, num_frames, channels, height, width] convert to [batch_size, channels, num_frames, height, width]
        x = x.transpose(1, 2)
        return self.backbone(x)
    
class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3), 
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )

        for param in self.conv_layers.parameters():
            param.requires_grad = False
        
        self.projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    
    def forward(self, x):
        x = x.suqeeze(1)
        features = self.conv_layers(x)
        return self.projection(features.sequeeze(-1))
    

class MultimodalSentimentModel(nn.Module):
    def __init__(self):
        super(MultimodalSentimentModel, self).__init__()
        self.text_model = TextModel()
        self.video_model = VideoEncoder()
        self.audio_model = AudioEncoder()

        # Fusion Layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 * 3, 256), 
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Classification Heads
        self.emotion_classifier = nn.Sequential(
            nn.Linear(256, 64), 
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 7)
        )

        self.sentiment_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)
        )
    
    def forward(self, text_input, video_input, audio_input):
        text_features = self.text_model(text_input['input_ids'], text_input['attention_mask'])
        video_features = self.video_model(video_input)
        audio_features = self.audio_model(audio_input)

        # Concatenate features
        features = torch.cat([text_features, video_features, audio_features], dim=1)
        fused_features = self.fusion_layer(features)

        emotion_logits = self.emotion_classifier(fused_features)
        sentiment_logits = self.sentiment_classifier(fused_features)

        return {
            'emotion': emotion_logits,
            'sentiment': sentiment_logits
        }

def compute_class_weights(dataset):
    emotion_counts = torch.zeros(7)
    sentiment_counts = torch.zeros(3)
    skipped = 0 
    total = len(dataset) 

    print("Computing class weights")

    for i in range(total):
        sample = dataset[i]
        
        if sample == None:
            skipped += 1
        
        emotion_counts[sample['emotion']] += 1
        sentiment_counts[sample['sentiment']] += 1
    
    valid = total - skipped
    print(f"Skipped {skipped} samples out of {total}")

    print("\nClass distribution:")
    print("Emotion:")
    emotion_map = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'joy', 4: 'neutral', 5: 'sadness', 6: 'surprise'}
    for i, count in enumerate(emotion_counts):
        print(f"{emotion_map[i]}: {count}")
    
    print("\nSentiment:")
    sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    for i, count in enumerate(sentiment_counts):
        print(f"{sentiment_map[i]}: {count}")
    
    # Calculate class weights
    emotion_weights = 1.0 / emotion_counts
    sentiment_weights = 1.0 / sentiment_counts

    # Normalize weights
    emotion_weights /= emotion_weights.sum()
    sentiment_weights /= sentiment_weights.sum()

    return emotion_weights, sentiment_weights
 

class MultimodalTrainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model 
        self.train_loader = train_loader
        self.val_loader = val_loader

        # log dataset size
        train_size = len(self.train_loader.dataset)
        val_size = len(self.val_loader.dataset)
        print("\nDataset sizes:")
        print(f"Train: {train_size} samples")
        print(f"Validation: {val_size} samples")
        print(f"Batch size: {len(self.train_loader.batch_size)}")

        timestamp = datetime.now().strftime("%b%d_%H-%M-%S")
        base_dir = 'opt/ml/output/tensorboard' if 'SM_MODEL_DIR' in os.environ else 'runs'
        log_dir = f"{base_dir}/run_{timestamp}"

        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0
        self.current_train_loss = None
        emotion_weights, sentiment_weights = compute_class_weights(train_loader.dataset)

        device = next(self.model.parameters()).device
        self.emotion_weights = emotion_weights.to(device)
        self.sentiment_weights = sentiment_weights.to(device)


        # Very high :1, high: 0.1 - 0.01, low: 1e-4, very low: 1e-5
        self.optimizer = torch.optim.Adam([
            {'params': model.text_encoder.parameters(), 'lr': 8e-6},
            {'params': model.video_encoder.parameters(), 'lr': 8e-5},
            {'params': model.audio_encoder.parameters(), 'lr': 8e-5},
            {'params': model.fusion_layer.parameters(), 'lr': 5e-4},
            {'params': model.emotion_classifier.parameters(), 'lr': 5e-4},
            {'params': model.sentiment_classifier.parameters(), 'lr': 5e-4}
        ], weight_decay=1e-5)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=2)

        self.emotion_criterion = nn.CrossEntropyLoss(
            label_smoothing=0.05,
            weight=self.emotion_weights
        )

        self.sentiment_criterion = nn.CrossEntropyLoss(
            label_smoothing=0.05,
            weight=self.sentiment_weights
        )

        def log_metrics(self, losses, metrics=None, phase='train'):
            if phase == 'train':
                self.current_train_loss = losses
            else:
                self.writer.add_scalar(
                    'loss/total/train', self.current_train_loss['total'], self.global_step
                )
                self.writer.add_scalar(
                    'loss/emotion/train', self.current_train_loss['emotion'], self.global_step
                )
                self.writer.add_scalar(
                    'loss/sentiment/train', self.current_train_loss['sentiment'], self.global_step
                )

                self.writer.add_scalar(
                    'loss/total/val', losses['total'], self.global_step
                )
                self.writer.add_scalar(
                    'loss/emotion/val', losses['emotion'], self.global_step
                )
                self.writer.add_scalar(
                    'loss/sentiment/val', losses['sentiment'], self.global_step
                )
            
            if metrics:
                self.writer.add_scalar(
                    f'{phase}/emotion_precision', metrics['emotion_precision'], self.global_step
                )
                self.writer.add_scalar(
                    f'{phase}/sentiment_precision', metrics['sentiment_precision'], self.global_step
                )
                self.writer.add_scalar(
                    f'{phase}/emotion_accuracy', metrics['emotion_accuracy'], self.global_step
                )
                self.writer.add_scalar(
                    f'{phase}/sentiment_accuracy', metrics['sentiment_accuracy'], self.global_step
                )
                
                
        def train_epoch(self):
            self.model.train()
            running_loss = {'total': 0.0, 'emotion': 0.0, 'sentiment': 0.0}

            for batch in self.train_loader:
                device = next(self.model.parameters()).device
                text_inputs = {k: v.to(device) for k, v in batch['text_inputs'].items()}
                video_frames = batch['video_frames'].to(device)
                audio_features = batch['audio_features'].to(device)
                emotion_labels = batch['emotion'].to(device)
                sentiment_labels = batch['sentiment'].to(device)

                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(text_inputs, video_frames, audio_features)

                # Calculate loss
                emotion_loss = self.emotion_criterion(outputs['emotion'], emotion_labels)
                sentiment_loss = self.sentiment_criterion(outputs['sentiment'], sentiment_labels)

                total_loss = emotion_loss + sentiment_loss

                # Backward pass
                total_loss.backward()

                # gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Update weights
                self.optimizer.step()
                
                # Log loss
                running_loss['total'] += total_loss.item()
                running_loss['emotion'] += emotion_loss.item()
                running_loss['sentiment'] += sentiment_loss.item()

                self.log_metrics(
                    {
                        'total': total_loss.item(), 
                        'emotion': emotion_loss.item(),
                        'sentiment': sentiment_loss.item()
                    }
                )
                
                self.global_step += 1

            return {k: v / len(self.train_loader) for k, v in running_loss.items()} 
        
        def evaluate(self, data_loader, phase="val"):
            self.model.eval()
            losses = {'total': 0.0, 'emotion': 0.0, 'sentiment': 0.0}
            all_emotion_labels = []
            all_sentiment_labels = []
            all_emotion_preds = []
            all_sentiment_preds = []

            with torch.inference_mode():
                for batch in data_loader:
                    device = next(self.model.parameters()).device
                    text_inputs = {k: v.to(device) for k, v in batch['text_inputs'].items()}
                    video_frames = batch['video_frames'].to(device)
                    audio_features = batch['audio_features'].to(device)
                    emotion_labels = batch['emotion'].to(device)
                    sentiment_labels = batch['sentiment'].to(device)

                    # Forward pass
                    outputs = self.model(text_inputs, video_frames, audio_features)

                    # Calculate loss
                    emotion_loss = self.emotion_criterion(outputs['emotion'], emotion_labels)
                    sentiment_loss = self.sentiment_criterion(outputs['sentiment'], sentiment_labels)
                    total_loss = emotion_loss + sentiment_loss

                    # Log loss
                    all_emotion_preds.extend(outputs['emotion'].argmax(dim=1).cpu().numpy())
                    all_emotion_labels.extend(emotion_labels.cpu().numpy())    
                    all_sentiment_preds.extend(outputs['sentiment'].argmax(dim=1).cpu().numpy())
                    all_sentiment_labels.extend(sentiment_labels.cpu().numpy())

                    losses['total'] += total_loss.item()
                    losses['emotion'] += emotion_loss.item()
                    losses['sentiment'] += sentiment_loss.item()
            
            avg_loss = {k: v / len(data_loader) for k, v in losses.items()}

            # Compute metrics -> Precision, Accuracy
            emotion_precision = precision_score(all_emotion_labels, all_emotion_preds, average='weighted')
            sentiment_precision = precision_score(all_sentiment_labels, all_sentiment_preds, average='weighted')
            emotion_accuracy = accuracy_score(all_emotion_labels, all_emotion_preds)
            sentiment_accuracy = accuracy_score(all_sentiment_labels, all_sentiment_preds)

            self.log_metrics(avg_loss, {
                'emotion_precision': emotion_precision,
                'sentiment_precision': sentiment_precision,
                'emotion_accuracy': emotion_accuracy,
                'sentiment_accuracy': sentiment_accuracy
            }, phase=phase)

            if phase == 'val':
                self.scheduler.step(avg_loss['total'])

            return avg_loss, {
                'emotion_precision': emotion_precision,
                'sentiment_precision': sentiment_precision,
                'emotion_accuracy': emotion_accuracy,
                'sentiment_accuracy': sentiment_accuracy
            }



