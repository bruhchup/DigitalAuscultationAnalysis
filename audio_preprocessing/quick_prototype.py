"""
Quick prototype to test if the approach works
Uses minimal preprocessing - just proof of concept
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
import random


def load_random_samples(data_dir, samples_per_class=100):
    """
    Load random samples from each class
    """
    data_path = Path(data_dir)
    
    X = []
    y = []
    
    for category in ['normal', 'crackle', 'wheeze']:
        cat_dir = data_path / category
        wav_files = list(cat_dir.glob('*.wav'))
        
        # Take random sample
        sample_files = random.sample(wav_files, min(samples_per_class, len(wav_files)))
        
        print(f"Loading {len(sample_files)} samples from {category}...")
        
        for wav_file in sample_files:
            try:
                # Load audio (quick, no resampling)
                audio, sr = librosa.load(wav_file, sr=None, duration=5.0)
                
                # Simple: just take first 5 seconds
                if len(audio) < sr * 5:
                    # Pad with zeros if too short
                    audio = np.pad(audio, (0, int(sr * 5 - len(audio))))
                else:
                    # Trim if too long
                    audio = audio[:int(sr * 5)]
                
                # Convert to mel spectrogram (basic, no optimization)
                mel_spec = librosa.feature.melspectrogram(
                    y=audio,
                    sr=sr,
                    n_mels=128,
                    fmax=2000
                )
                
                # Convert to dB
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                
                X.append(mel_spec_db)
                y.append(category)
                
            except Exception as e:
                print(f"Error loading {wav_file.name}: {e}")
                continue
    
    return np.array(X), np.array(y)


def build_quick_model(input_shape, num_classes=3):
    """
    Quick and dirty CNN - just to test
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def visualize_sample_spectrograms(X, y, num_samples=3):
    """
    Visualize sample spectrograms from each class
    """
    fig, axes = plt.subplots(3, num_samples, figsize=(15, 8))
    
    categories = ['normal', 'crackle', 'wheeze']
    
    for i, category in enumerate(categories):
        # Find samples of this category
        indices = np.where(y == category)[0]
        sample_indices = random.sample(list(indices), min(num_samples, len(indices)))
        
        for j, idx in enumerate(sample_indices):
            ax = axes[i, j] if num_samples > 1 else axes[i]
            
            librosa.display.specshow(
                X[idx],
                ax=ax,
                y_axis='mel',
                x_axis='time',
                cmap='viridis'
            )
            
            if j == 0:
                ax.set_ylabel(category.capitalize(), fontsize=12)
            else:
                ax.set_ylabel('')
            
            if i == 0:
                ax.set_title(f'Sample {j+1}', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('quick_prototype_spectrograms.png', dpi=150)
    print("\nSUCCESS: Sample spectrograms saved: quick_prototype_spectrograms.png")
    plt.close()


if __name__ == "__main__":
    print("="*70)
    print("QUICK PROTOTYPE - PROOF OF CONCEPT")
    print("="*70)
    print("\nThis is a rapid test to see if the approach works.")
    print("We'll use minimal preprocessing and a small dataset.\n")
    
    # Get script directory
    script_dir = Path(__file__).parent
    DATA_DIR = script_dir.parent / "data" / "categorized_cycles_3class"
    
    # Check if data exists
    if not DATA_DIR.exists():
        print(f"Error: Data directory not found: {DATA_DIR}")
        print("Please run prepare_3class_dataset.py first!")
        exit(1)
    
    # Step 1: Load data
    print("Step 1: Loading random samples...")
    X, y = load_random_samples(DATA_DIR, samples_per_class=100)
    
    print(f"\nLoaded {len(X)} samples")
    print(f"Spectrogram shape: {X[0].shape}")
    
    # Visualize samples
    print("\nStep 2: Visualizing samples...")
    visualize_sample_spectrograms(X, y, num_samples=3)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print("\nClass distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cat, count in zip(unique, counts):
        print(f"  {cat}: {count}")
    
    # Step 3: Split data
    print("\nStep 3: Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )
    
    # Add channel dimension for CNN
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Step 4: Build model
    print("\nStep 4: Building quick model...")
    model = build_quick_model(X_train[0].shape, num_classes=3)
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel architecture:")
    model.summary()
    
    # Step 5: Train
    print("\nStep 5: Training (quick - just 10 epochs)...")
    print("This will take a few minutes...\n")
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=10,
        batch_size=32,
        verbose=1
    )
    
    # Step 6: Evaluate
    print("\nStep 6: Evaluating...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print("\n" + "="*70)
    print("QUICK PROTOTYPE RESULTS")
    print("="*70)
    print(f"\nTest Accuracy: {test_accuracy*100:.2f}%")
    print(f"Random Guessing: 33.33%")
    
    if test_accuracy > 0.333:
        improvement = (test_accuracy/0.3333 - 1)*100
        print(f"Improvement: {improvement:.1f}% better than random")
    else:
        print("WARNING: Not better than random guessing!")
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\nConfusion Matrix:")
    print("(rows=actual, columns=predicted)")
    cm = confusion_matrix(y_test, y_pred_classes)
    
    # Print with labels
    print(f"\n{'':12} {le.classes_[0]:>10} {le.classes_[1]:>10} {le.classes_[2]:>10}")
    for i, category in enumerate(le.classes_):
        print(f"{category:12} {cm[i][0]:>10} {cm[i][1]:>10} {cm[i][2]:>10}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=le.classes_))
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train', marker='o')
    plt.plot(history.history['val_accuracy'], label='Validation', marker='s')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train', marker='o')
    plt.plot(history.history['val_loss'], label='Validation', marker='s')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quick_prototype_training.png', dpi=150)
    print("\nSUCCESS: Training curves saved: quick_prototype_training.png")
    plt.close()
    
    # Conclusion
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    if test_accuracy > 0.50:
        print("\nEXCELLENT! The concept works very well!")
        print(f"With minimal preprocessing, we achieved {test_accuracy*100:.1f}% accuracy.")
    elif test_accuracy > 0.40:
        print("\nGOOD! The concept is validated!")
        print(f"With minimal preprocessing, we achieved {test_accuracy*100:.1f}% accuracy.")
    elif test_accuracy > 0.35:
        print("\nREASONABLE - The model is learning something!")
        print(f"We achieved {test_accuracy*100:.1f}% accuracy.")
    else:
        print("\nNEEDS WORK - Model is struggling")
        print(f"Only achieved {test_accuracy*100:.1f}% accuracy.")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    
    if test_accuracy > 0.35:
        print("\nCONCEPT VALIDATED - Proceed with full pipeline!")
        print("\nWhat to do next:")
        print("1. Implement full preprocessing pipeline")
        print("   - Standardize all cycles to 8 seconds")
        print("   - Apply bandpass filtering (50-2000 Hz)")
        print("   - Proper normalization")
        print("")
        print("2. Use patient-level train/test split")
        print("   - Prevent data leakage")
        print("   - More realistic evaluation")
        print("")
        print("3. Implement data augmentation")
        print("   - Time stretching")
        print("   - Pitch shifting")
        print("   - Noise injection")
        print("")
        print("4. Train with full dataset")
        print("   - Use all 6,392 cycles")
        print("   - More epochs (30-50)")
        print("   - Better architecture")
        print("")
        print("Expected improvement: +10-20% accuracy with proper pipeline")
        print(f"Target: 55-65% test accuracy")
    else:
        print("\nNeed to investigate:")
        print("- Check spectrograms visually")
        print("- Try different preprocessing")
        print("- Verify data loading is correct")
        print("- Try simpler model first")
    
    print("\n" + "="*70)
    print("PROTOTYPE COMPLETE!")
    print("="*70)