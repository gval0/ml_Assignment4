# ml_Assignment4

## პროექტის აღწერა

ამ პროექტში შესრულებულია **Challenges in Representation Learning: Facial Expression Recognition Challenge** Kaggle ქომფეთიშენის ამოცანა. პროექტის მიზანია სხვადასხვა ნეირონული ქსელის არქიტექტურების გატესტვა.

## 🔗 ლინკები

- **GitHub Repository**: [https://github.com/gval0/ml_Assignment4](https://github.com/gval0/ml_Assignment4)
- **Wandb Project**: [facial-expression-recognition](https://wandb.ai/gval0/facial-expression-recognition)
- **Kaggle Competition**: [Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)

## 🏗️ პროექტის სტრუქტურა

```
ml_Assignment4/
├── src/
│   ├── models/
│   │   ├── simple_cnn.py      # მარტივი CNN
│   │   ├── deeper_cnn.py      # ღრმა CNN
│   │   └── resnet_variants.py # ResNet ვარიანტები
│   ├── training/
│   │   ├── trainer.py         # სასწავლო კლასი
│   │   └── utils.py          # დამხმარე ფუნქციები
│   ├── evaluation/
│   │   └── metrics.py        # შეფასების მეტრიკები
│   └── data_loader.py        # მონაცემების ჩატვირთვა
├── notebooks/
│   ├── 01_data_exploration.ipynb    # მონაცემების ანალიზი
│   ├── 02_baseline_models.ipynb     # საბაზისო მოდელები
│   ├── 03_model_iterations.ipynb    # მოდელის იტერაციები
│   └── 04_final_analysis.ipynb      # საბოლოო ანალიზი
├── experiments/                     # შენახული მოდელები
└── README.md
```

## 🧪 ჩატარებული ექსპერიმენტები

### სერია 1: მონაცემების დამუშავება (Data Augmentation)
- **Exp1a**: საბაზისო მოდელი no augmentation → **55.67%** სიზუსტე
- **Exp1b**: მოდელი with augmentation → **57.06%** სიზუსტე
- **დასკვნა**: მონაცემების ზრდამ გააუმჯობესა შედეგი 1.4%-ით

### სერია 2: არქიტექტურების შედარება
- **SimpleCNN**: 4.74M პარამეტრი → **57.59%** სიზუსტე
- **DeepCNN**: 1.70M პარამეტრი → **25.83%** სიზუსტე (underfitting)
- **ResNetFER**: 2.78M პარამეტრი → **59.31%** ზუსტობა ⭐

### სერია 3: სწავლის ტემპის ოპტიმიზაცია (Learning Rate)
ResNetFER-ზე სხვადასხვა LR-ების ტესტირება:
- **1e-4**: 46.47% (ნელი სწავლება)
- **1e-3**: 57.13% 
- **1e-2**: 58.96% ⭐

### სერია 4: ოპტიმაიზერების შედარება
- **Adam**: 56.99% სიზუსტე
- **SGD**: 48.16% სიზუსტე
- **დასკვნა**: Adam უკეთესი შედეგი აჩვენა

### სერია 5: რეგულარიზაცია (Dropout)
- **Dropout 0.3**: 58.28% ⭐
- **Dropout 0.5**: 57.15%  
- **Dropout 0.7**: 57.45%

## 🏆 საბოლოო შედეგები

**საუკეთესო მოდელი**:
- **არქიტექტურა**: ResNetFER
- **ზუსტობა**: **58.28%**
- **პარამეტრები**: 2,778,311
- **ჰიპერპარამეტრები**:
  - Learning Rate: 0.01
  - Optimizer: Adam
  - Dropout: 0.3
  - Data Augmentation: True

## 📈 ძირითადი დაკვირვებები

### 1. **Overfitting-ის ანალიზი**
- ResNetFER-მა აჩვენა ძლიერი overfitting (Train: 95% vs Val: 58%)
- Dropout-მა დაეხმარა overfitting-ის შემცირებაში
- Data Augmentation-მა ასევე შეამცირა overfitting

### 2. **Underfitting-ის ანალიზი**  
- DeepCNN აჩვენებდა underfitting-ს (25% ზუსტობა)
- მიზეზი: ძალიან ღრმა არქიტექტურა მცირე მონაცემებისთვის
- Gradient vanishing პრობლემა

### 3. **მოდელის ზომისა და შესრულების კავშირი**
- SimpleCNN (4.7M): 57.59%
- ResNetFER (2.8M): 59.31%
- **დასკვნა**: მეტი პარამეტრი ≠ უკეთესი შედეგი

## 📊 Wandb Tracking

ყველა ექსპერიმენტი ლოგირებულია Wandb-ზე:
- **პროექტი**: `facial-expression-recognition`
- **სულ Runs**: 14+ ექსპერიმენტი
- **მეტრიკები**: accuracy, loss, learning curves, confusion matrix

### ლოგირებული მეტრიკები:
- Train/Validation Accuracy & Loss
- Train-Validation Gap (overfitting ინდიკატორი)
- Model Parameters რაოდენობა
- Training Time
- Hyperparameters

## 📁 ფაილების სტრუქტურა

| ფაილი | აღწერა |
|--------|---------|
| `01_data_exploration.ipynb` | მონაცემების ანალიზი და ვიზუალიზაცია |
| `02_baseline_models.ipynb` | საბაზისო მოდელების იმპლემენტაცია |
| `03_model_iterations.ipynb` | ყველა ექსპერიმენტის სისტემატური ჩატარება |
| `04_final_analysis.ipynb` | საბოლოო შედეგების ანალიზი |

## 🎓 ძირითადი დასკვნები

1. **ResNet არქიტექტურა** უკეთესია მარტივ CNN-ზე FER ამოცანისთვის
2. **Data Augmentation** მნიშვნელოვნად ამცირებს overfitting-ს
3. **Learning Rate 0.01** ოპტიმალურია Adam optimizer-თან
4. **Dropout 0.3** ოპტიმალური კომპრომისია accuracy-სა და regularization-ს შორის
5. **Overfitting** ძირითადი პრობლემაა - მოდელი ზედმეტად სწავლობს სასწავლო მონაცემებს