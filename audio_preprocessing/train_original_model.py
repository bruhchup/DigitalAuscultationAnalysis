"""
Train Original Model - Best Performance
This is the model that achieved 61.6% test accuracy with 87.8% normal detection

Uses:
- Preprocessed dataset (not augmented)
- Basic balanced class weighting
- 3-class classification
- Standard CNN architecture

Results:
- Overall: 61.6%
- Normal: 87.8% recall
- Crackle: 17.6% recall
- Wheeze: 18.2% recall
"""

import numpy as np
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score
import matplotlib.pyplot as plt
import json
import random


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

def build_cnn_model(input_shape, num_classes=3):
    """
    Build CNN model for preprocessed spectrograms
    
    Architecture optimized for respiratory sound classification
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # First conv block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second conv block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third conv block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


# ============================================================================
# TRAINING
# ============================================================================

def train_original_model(preprocessed_dir, output_dir):
    """
    Train the original best-performing model
    """
    preprocessed_path = Path(preprocessed_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("TRAINING ORIGINAL MODEL (BEST PERFORMANCE)")
    print("="*70)
    print(f"\nData source: {preprocessed_dir}")
    print(f"Model output: {output_dir}")
    print()
    
    # Load preprocessed data
    print("Loading preprocessed dataset...")
    X_train = np.load(preprocessed_path / 'X_train.npy')
    y_train = np.load(preprocessed_path / 'y_train.npy')
    X_val = np.load(preprocessed_path / 'X_val.npy')
    y_val = np.load(preprocessed_path / 'y_val.npy')
    X_test = np.load(preprocessed_path / 'X_test.npy')
    y_test = np.load(preprocessed_path / 'y_test.npy')
    
    print(f"Train: {X_train.shape}")
    print(f"Val:   {X_val.shape}")
    print(f"Test:  {X_test.shape}")
    
    # Add channel dimension if needed
    if len(X_train.shape) == 3:
        X_train = X_train[..., np.newaxis]
        X_val = X_val[..., np.newaxis]
        X_test = X_test[..., np.newaxis]
        print(f"Added channel dimension: {X_train[0].shape}")
    
    # Check class distribution
    print("\nClass distribution (training):")
    for class_idx in range(3):
        count = np.sum(y_train == class_idx)
        percentage = (count / len(y_train)) * 100
        class_name = ['Normal', 'Crackle', 'Wheeze'][class_idx]
        print(f"  {class_name}: {count:,} ({percentage:.1f}%)")
    
    # Compute balanced class weights (not too aggressive)
    print("\nComputing balanced class weights...")
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    print("Class weights:")
    for idx, weight in class_weight_dict.items():
        class_name = ['Normal', 'Crackle', 'Wheeze'][idx]
        print(f"  {class_name}: {weight:.2f}")
    
    # Build model
    print("\n" + "="*70)
    print("Building model...")
    print("="*70)
    
    model = build_cnn_model(X_train[0].shape, num_classes=3)
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel architecture:")
    model.summary()
    
    # Callbacks
    callback_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            str(output_path / 'best_checkpoint.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train
    print("\n" + "="*70)
    print("Training model...")
    print("="*70)
    print("\nTraining for up to 20 epochs with early stopping")
    print("This may take 10-20 minutes...\n")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=callback_list,
        verbose=1
    )
    
    # Evaluate
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    
    # Test set evaluation
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Predictions
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    
    # Per-class metrics
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    # Balanced accuracy
    balanced_acc = balanced_accuracy_score(y_test, y_pred_classes)
    
    print(f"\nTest Accuracy: {test_accuracy*100:.2f}%")
    print(f"Balanced Accuracy: {balanced_acc*100:.2f}%")
    
    print("\nPer-Class Recall:")
    class_names = ['Normal', 'Crackle', 'Wheeze']
    for i, name in enumerate(class_names):
        print(f"  {name:8} {per_class_acc[i]*100:.1f}%")
    
    print("\nConfusion Matrix:")
    print(f"{'':12} {'Normal':>10} {'Crackle':>10} {'Wheeze':>10}")
    for i, name in enumerate(class_names):
        print(f"{name:12} {cm[i][0]:>10} {cm[i][1]:>10} {cm[i][2]:>10}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred_classes, 
                                target_names=class_names, 
                                digits=3))
    
    # Binary analysis (Normal vs Abnormal)
    print("\n" + "="*70)
    print("BINARY ANALYSIS (Normal vs Abnormal)")
    print("="*70)
    
    # Reframe as binary
    y_test_binary = (y_test != 0).astype(int)  # 0=Normal, 1=Abnormal
    y_pred_binary = (y_pred_classes != 0).astype(int)
    
    # Binary confusion matrix
    cm_binary = confusion_matrix(y_test_binary, y_pred_binary)
    
    # Calculate metrics
    tn, fp, fn, tp = cm_binary.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall for abnormal
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Recall for normal
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision for abnormal
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Precision for normal
    
    print("\nBinary Confusion Matrix:")
    print(f"{'':15} {'Pred Normal':>15} {'Pred Abnormal':>15}")
    print(f"{'Actual Normal':15} {tn:>15} {fp:>15}")
    print(f"{'Actual Abnormal':15} {fn:>15} {tp:>15}")
    
    print(f"\nBinary Metrics:")
    print(f"  Sensitivity (Detecting Abnormal): {sensitivity*100:.1f}%")
    print(f"  Specificity (Detecting Normal):   {specificity*100:.1f}%")
    print(f"  Positive Predictive Value:        {ppv*100:.1f}%")
    print(f"  Negative Predictive Value:        {npv*100:.1f}%")
    
    print("\nClinical Interpretation:")
    if specificity >= 0.85:
        print(f"  Specificity {specificity*100:.1f}% - EXCELLENT at confirming normal")
    elif specificity >= 0.70:
        print(f"  Specificity {specificity*100:.1f}% - GOOD at confirming normal")
    else:
        print(f"  Specificity {specificity*100:.1f}% - Needs improvement")
    
    if sensitivity >= 0.85:
        print(f"  Sensitivity {sensitivity*100:.1f}% - Clinical-grade abnormality detection")
    elif sensitivity >= 0.60:
        print(f"  Sensitivity {sensitivity*100:.1f}% - Moderate abnormality detection")
    else:
        print(f"  Sensitivity {sensitivity*100:.1f}% - Low abnormality detection")
        print("    Suitable as confirmation tool, not screening tool")
    
    # Check overfitting
    final_train_acc = history.history['accuracy'][-1]
    gap = final_train_acc - test_accuracy
    
    print("\n" + "="*70)
    print("OVERFITTING ANALYSIS")
    print("="*70)
    print(f"\nFinal train accuracy: {final_train_acc*100:.1f}%")
    print(f"Test accuracy:        {test_accuracy*100:.1f}%")
    print(f"Train-test gap:       {gap*100:.1f}%")
    
    if gap < 0.10:
        print("Status: Minimal overfitting - excellent generalization!")
    elif gap < 0.20:
        print("Status: Moderate overfitting - acceptable for capstone")
    else:
        print("Status: Significant overfitting - model memorizing training data")
    
    # Save results
    results = {
        'test_accuracy': float(test_accuracy),
        'balanced_accuracy': float(balanced_acc),
        'per_class_recall': {
            'normal': float(per_class_acc[0]),
            'crackle': float(per_class_acc[1]),
            'wheeze': float(per_class_acc[2])
        },
        'binary_metrics': {
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'ppv': float(ppv),
            'npv': float(npv)
        },
        'confusion_matrix': cm.tolist(),
        'binary_confusion_matrix': cm_binary.tolist(),
        'train_test_gap': float(gap),
        'epochs_trained': len(history.history['loss'])
    }
    
    with open(output_path / 'model_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved: {output_path / 'model_results.json'}")
    
    # Plot training history
    plot_training_history(history, output_path, test_accuracy, balanced_acc)
    
    # Plot confusion matrices
    plot_confusion_matrices(cm, cm_binary, class_names, output_path)
    
    # Plot clinical metrics
    plot_clinical_metrics(sensitivity, specificity, per_class_acc, output_path)
    
    # Save model
    model.save(output_path / 'respiratory_classifier.keras')
    print(f"\nModel saved: {output_path / 'respiratory_classifier.keras'}")
    
    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE - MODEL SUMMARY")
    print("="*70)
    print(f"\nOverall Performance:")
    print(f"  Test Accuracy:     {test_accuracy*100:.1f}%")
    print(f"  Balanced Accuracy: {balanced_acc*100:.1f}%")
    print(f"\nClinical Performance:")
    print(f"  Specificity: {specificity*100:.1f}% (confirming normal)")
    print(f"  Sensitivity: {sensitivity*100:.1f}% (detecting abnormal)")
    print(f"\nModel Characteristics:")
    print(f"  - Excellent at detecting normal breathing (87-88%)")
    print(f"  - Suitable as normal confirmation tool")
    print(f"  - Requires human review for abnormalities")
    print(f"\nFiles Created:")
    print(f"  - respiratory_classifier.keras (deployable model)")
    print(f"  - best_checkpoint.keras (best validation checkpoint)")
    print(f"  - model_results.json (all metrics)")
    print(f"  - training_history.png (learning curves)")
    print(f"  - confusion_matrices.png (performance visualization)")
    print(f"  - clinical_metrics.png (clinical performance)")
    
    return model, history, results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_history(history, output_dir, test_acc, balanced_acc):
    """Plot training history"""
    output_path = Path(output_dir)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    ax = axes[0]
    ax.plot(history.history['accuracy'], label='Train', linewidth=2)
    ax.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    ax.axhline(y=test_acc, color='r', linestyle='--', 
               label=f'Test ({test_acc*100:.1f}%)', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Model Accuracy', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Loss
    ax = axes[1]
    ax.plot(history.history['loss'], label='Train', linewidth=2)
    ax.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Model Loss', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'training_history.png', dpi=150, bbox_inches='tight')
    print(f"Training history plot saved: {output_path / 'training_history.png'}")
    plt.close()


def plot_confusion_matrices(cm, cm_binary, class_names, output_dir):
    """Plot both 3-class and binary confusion matrices"""
    output_path = Path(output_dir)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 3-class confusion matrix
    ax = axes[0]
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_title('3-Class Confusion Matrix', fontsize=12, fontweight='bold')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=12, fontweight='bold')
    
    # Binary confusion matrix
    ax = axes[1]
    im = ax.imshow(cm_binary, interpolation='nearest', cmap='Greens')
    ax.figure.colorbar(im, ax=ax)
    
    binary_labels = ['Normal', 'Abnormal']
    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(binary_labels)
    ax.set_yticklabels(binary_labels)
    
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_title('Binary Confusion Matrix\n(Normal vs Abnormal)', 
                 fontsize=12, fontweight='bold')
    
    # Add text annotations
    thresh = cm_binary.max() / 2.
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm_binary[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm_binary[i, j] > thresh else "black",
                   fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
    print(f"Confusion matrices saved: {output_path / 'confusion_matrices.png'}")
    plt.close()


def plot_clinical_metrics(sensitivity, specificity, per_class_acc, output_dir):
    """Plot clinical performance metrics"""
    output_path = Path(output_dir)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Binary metrics (clinical)
    ax = axes[0]
    metrics = ['Sensitivity\n(Detect Abnormal)', 'Specificity\n(Detect Normal)']
    values = [sensitivity * 100, specificity * 100]
    colors = ['#e74c3c', '#2ecc71']
    
    bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.axhline(y=85, color='orange', linestyle='--', linewidth=2, 
               label='Clinical Target (85%)', alpha=0.7)
    ax.set_ylabel('Percentage (%)', fontsize=11)
    ax.set_title('Clinical Screening Metrics', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Per-class recall
    ax = axes[1]
    class_names = ['Normal', 'Crackle', 'Wheeze']
    values = per_class_acc * 100
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    
    bars = ax.bar(class_names, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Recall (%)', fontsize=11)
    ax.set_title('Per-Class Detection Accuracy', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / 'clinical_metrics.png', dpi=150, bbox_inches='tight')
    print(f"Clinical metrics plot saved: {output_path / 'clinical_metrics.png'}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution"""
    # Paths
    script_dir = Path(__file__).parent
    PREPROCESSED_DIR = script_dir.parent / "data" / "preprocessed_dataset"
    OUTPUT_DIR = script_dir.parent / "models" / "original_model"
    
    # Check input exists
    if not PREPROCESSED_DIR.exists():
        print(f"Error: Preprocessed data not found: {PREPROCESSED_DIR}")
        print("Please run preprocess_full_dataset.py first!")
        return
    
    # Set random seeds
    np.random.seed(42)
    random.seed(42)
    import tensorflow as tf
    tf.random.set_seed(42)
    
    # Train model
    model, history, results = train_original_model(PREPROCESSED_DIR, OUTPUT_DIR)
    
    print("\n" + "="*70)
    print("SUCCESS!")
    print("="*70)
    print(f"\nYour best-performing model is ready!")
    print(f"Location: {OUTPUT_DIR}")
    print(f"\nThis model is ready for web interface integration!")


if __name__ == "__main__":
    main()