# Iris Classifier

Tento projekt obsahuje kompletný kód na trénovanie a hodnotenie neurónovej siete na klasifikáciu dát o irisoch. Používa knižnice PyTorch na vytvorenie modelu a jeho vyhodnotenie. Dátový set obsahuje informácie o druhoch irisov a ich rozmeroch.

## Obsah

1. [Inštalácia a importovanie knižníc](#inštalácia-a-importovanie-knižníc)
2. [Načítanie a predspracovanie dát](#načítanie-a-predspracovanie-dát)
3. [Definícia modelu](#definícia-modelu)
4. [Trénovanie modelu](#trénovanie-modelu)
5. [Vyhodnotenie modelu](#vyhodnotenie-modelu)
6. [Predikcia a ukladanie výsledkov](#predikcia-a-ukladanie-výsledkov)

## Inštalácia a importovanie knižníc

Najprv je potrebné nainštalovať potrebné knižnice. Ak ešte nemáte nainštalované tieto knižnice, použite nasledujúce príkazy:

```bash
pip install torch torchvision pandas scikit-learn seaborn matplotlib
```
### Importujte potrebné knižnice do vášho Python skriptu:

```bash
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
```
## Načítanie a predspracovanie dát
1. Načítanie dát: Načítame dáta z CSV súboru obsahujúceho informácie o irisoch.
```bash
csv_file = "dataset/iris.csv"  # Cesta k CSV súboru
df = pd.read_csv(csv_file)  # Načítanie dát do DataFrame
```
2. Rozdelenie na features a labels: Oddelíme vstupné údaje (features) od výstupných (labels).
```bash
X = df.drop("species", axis=1).values  # Vstupné údaje
y = df["species"].values  # Výstupné štítky
```
3. Transformácia textu: Prevedieme textové štítky na číselné.
```bash
le = LabelEncoder()  # Inicializácia LabelEncoder
y = le.fit_transform(y)  # Transformácia textových štítkov na číselné
```
4. Rozdelenie na testovacie a trénovacie dáta: Rozdelíme dáta na trénovaciu a testovaciu množinu.
```bash
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Rozdelenie dát
```
5. Normalizácia dát: Normalizujeme dáta, aby mali nulový priemer a jednotkovú odchýlku.
```bash
scaler = StandardScaler()  # Inicializácia normalizátora
X_train_scaled = scaler.fit_transform(X_train)  # Normalizácia trénovacích dát
X_test_scaled = scaler.transform(X_test)  # Normalizácia testovacích dát

```
6. Vytvorenie tensorov: Prevedieme dáta na PyTorch tenzory.
```bash
X_train_tensor = torch.FloatTensor(X_train_scaled)  # Vytvorenie tenzora pre trénovacie dáta
X_test_tensor = torch.FloatTensor(X_test_scaled)  # Vytvorenie tenzora pre testovacie dáta
y_train_tensor = torch.LongTensor(y_train)  # Vytvorenie tenzora pre trénovacie štítky
y_test_tensor = torch.LongTensor(y_test)  # Vytvorenie tenzora pre testovacie štítky

```
7. Vytvorenie datasetu: Vytvoríme PyTorch dataset pre trénovanie a testovanie.
```bash
class IrisDataset(Dataset):
    def __init__(self, X_features, y_labels):
        self.X_features = X_features
        self.y_labels = y_labels
    
    def __len__(self):
        return len(self.y_labels)
    
    def __getitem__(self, idx):
        return self.X_features[idx], self.y_labels[idx]
```
8. Vytvorenie dataloaderu: Vytvoríme DataLoader pre trénovanie a testovanie.
```bash
train_dataset = IrisDataset(X_train_tensor, y_train_tensor)  # Trénovací dataset
test_dataset = IrisDataset(X_test_tensor, y_test_tensor)  # Testovací dataset
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Trénovací DataLoader
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)  # Testovací DataLoader
```

## Definícia modelu
Vytvoríme neurónovú sieť s viacerými skrytými vrstvami.
```bash
class IrisNetwork(nn.Module):
    def __init__(self):
        super(IrisNetwork, self).__init__()
        self.fc1 = nn.Linear(4, 128)  # Prvá plne prepojená vrstva
        self.fc2 = nn.Linear(128, 64)  # Druhá plne prepojená vrstva
        self.fc3 = nn.Linear(64, 32)   # Tretia plne prepojená vrstva
        self.fc4 = nn.Linear(32, 3)    # Výstupná vrstva pre 3 triedy (iris)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Aktivácia ReLU po prvej vrstve
        x = torch.relu(self.fc2(x))  # Aktivácia ReLU po druhej vrstve
        x = torch.relu(self.fc3(x))  # Aktivácia ReLU po tretej vrstve
        x = self.fc4(x)  # Výstupná vrstva
        return x

model = IrisNetwork()  # Inicializácia modelu
criterion = nn.CrossEntropyLoss()  # Kritérium pre klasifikáciu
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)  # Optimalizátor
```
## Trénovanie modelu
Funkcia na trénovanie modelu a vizualizáciu výsledkov.
```bash
def train(model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs):
    train_losses = []
    test_losses = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()  # Prepnutie modelu do tréningového režimu
        train_loss = 0.0
        for features, labels in train_dataloader:
            optimizer.zero_grad()  # Vymazanie gradientov
            outputs = model(features)  # Predikcia
            loss = criterion(outputs, labels)  # Výpočet straty
            loss.backward()  # Výpočet gradientov
            optimizer.step()  # Aktualizácia váh
            train_loss += loss.item()
            
        train_losses.append(train_loss / len(train_dataloader))  # Uloženie priemernej trénovacej straty
        
        model.eval()  # Prepnutie modelu do evaluačného režimu
        test_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        with torch.no_grad():  # Deaktivácia výpočtu gradientov
            for features, labels in test_dataloader:
                outputs = model(features)  # Predikcia
                loss = criterion(outputs, labels)  # Výpočet testovacej straty
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)  # Určenie predikovaných štítkov
                total += labels.size(0)  # Počet vzoriek
                correct += (predicted == labels).sum().item()  # Počet správne klasifikovaných vzoriek

                all_labels.extend(labels.numpy())  # Uloženie skutočných štítkov
                all_predictions.extend(predicted.numpy())  # Uloženie predikovaných štítkov

        test_losses.append(test_loss / len(test_dataloader))  # Uloženie priemernej testovacej straty
        test_accuracy = 100 * correct / total  # Výpočet presnosti
        test_accuracies.append(test_accuracy)  # Uloženie testovacej presnosti
        
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_loss/len(train_dataloader):.4f}, '
                  f'Test Loss: {test_loss/len(test_dataloader):.4f}, '
                  f'Test Accuracy: {test_accuracy:.2f}%')
        
        # Vizualizácia stratovosti (Loss)
        epochs = range(1, num_epochs + 1)
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train and Test Loss over Epochs')
        plt.legend()
    
        # Vizualizácia testovacej presnosti
        plt.subplot(1, 2, 2)
        plt.plot(epochs, test_accuracies, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Test Accuracy over Epochs')
        plt.legend()
        plt.show()
    
        # Matica zámien (Confusion Matrix)
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
    
        # ROC krivka (v prípade binárnej klasifikácie alebo prispôsobenia pre multiclass)
        # Ak je klasifikácia binárna alebo ak máš len dve triedy, môžeš priamo použiť tento kód:
        if len(le.classes_) == 2:
            all_labels_bin = label_binarize(all_labels, classes=[0, 1])
            all_predictions_bin = label_binarize(all_predictions, classes=[0, 1])
    
            fpr, tpr, _ = roc_curve(all_labels_bin, all_predictions_bin)
            roc_auc = auc(fpr, tpr)
    
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.show()

```
## Vyhodnotenie modelu
Po natrénovaní modelu je možné vykonať jeho vyhodnotenie na testovacích dátach. Tento krok nám umožní získať presnosť modelu na testovacej množine.
```bash
# Vyhodnotenie modelu na testovacích dátach
correct = 0
total = 0
with torch.no_grad():  # Deaktivácia výpočtu gradientov
    for features, labels in test_dataloader:
        outputs = model(features)  # Predikcia
        _, predicted = torch.max(outputs.data, 1)  # Určenie predikovaných štítkov
        total += labels.size(0)  # Počet vzoriek
        correct += (predicted == labels).sum().item()  # Počet správne klasifikovaných vzoriek

accuracy = 100 * correct / total  # Výpočet presnosti
print(f'Test Accuracy of the model on the test data: {accuracy:.2f}%')
```
## Predikcia a ukladanie výsledkov
Na konci môžeme vykonať predikciu na nových dátach a uložiť výsledky do CSV súboru.
```bash
import pandas as pd

# Predpokladajme, že máš testovacie dáta už spracované
predictions = []
true_labels = []

model.eval()  # Prepnutie modelu do evaluačného režimu
with torch.no_grad():  # Deaktivácia výpočtu gradientov
    for features, labels in test_dataloader:
        outputs = model(features)  # Predikcia
        _, predicted = torch.max(outputs.data, 1)  # Určenie predikovaných štítkov
        predictions.extend(predicted.numpy())  # Uloženie predikovaných štítkov
        true_labels.extend(labels.numpy())  # Uloženie skutočných štítkov

# Vytvor DataFrame a ulož do CSV
df = pd.DataFrame({'True Label': true_labels, 'Predicted Label': predictions})
df.to_csv('predictions.csv', index=False)  # Uloženie do CSV súboru
```

### Záver
Tento projekt vám ukázal, ako vytvoriť a natrénovať neurónovú sieť na klasifikáciu dát o irisoch pomocou PyTorch. Naučili sme sa, ako predspracovať dáta, trénovať model, vyhodnotiť jeho výkon a vykonať predikcie na nových dátach. Vizualizácie, ako sú grafy stratovosti, testovacej presnosti, matica zámien a ROC krivka, poskytujú hlbší pohľad na výkonnosť modelu.

Nezabudnite prispôsobiť kód vašim špecifickým potrebám a experimentovať s rôznymi nastaveniami modelu a hyperparametrami pre dosiahnutie najlepších výsledkov.













