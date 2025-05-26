import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_training_curves(trainer):
    """Plot training/validation curves"""
    # Implementation here

def analyze_dataset(csv_file):
    data = pd.read_csv(csv_file)
    emotion_counts = data['emotion'].value_counts().sort_index()
    emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
                     4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(emotion_counts)), emotion_counts.values)
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.title('Distribution of Emotions in Dataset')
    plt.xticks(range(len(emotion_counts)), 
               [emotion_labels[i] for i in emotion_counts.index], rotation=45)
    for bar, count in zip(bars, emotion_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50,
                f'{count}', ha='center', va='bottom')
    
    plt.tight_layout()
    wandb.log({"emotion_distribution": wandb.Image(plt)})
    plt.show()
    
    print("Dataset Analysis:")
    print(f"Total samples: {len(data)}")
    print(f"Number of classes: {len(emotion_counts)}")
    print("\nClass distribution:")
    for idx, count in emotion_counts.items():
        print(f"{emotion_labels[idx]}: {count} ({count/len(data)*100:.1f}%)")