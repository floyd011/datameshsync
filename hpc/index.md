---
layout: default
title: "Aktivna edukacija za akademski HPC"
---

# Aktivna edukacija — detaljna razrada za akademski HPC #


SLOJ 1 — “Brzi start” (cilj: 15 minuta do prvog GPU joba)
Filozofija ovog sloja: nula teorije, maksimum copy-paste koda koji radi. Korisnik treba da doživi uspeh pre nego što shvati zašto nešto funkcioniše.

1.1 Prva stvar koju svaki korisnik mora da nauči — verifikacija
Pre bilo kakvog koda, korisnik mora znati kako da proveri da li GPU uopšte “vidi”:

**Na login/compute nodu — provera da li sistem vidi GPU**

```bash
nvidia-smi
```

**Očekivani izlaz (primer):**
```bash
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA A100-SXM...  On   | 00000000:07:00.0 Off |                    0 |
| N/A   30C    P0    52W / 400W |      0MiB / 81920MiB |      0%      Default |
+-----------------------------------------------------------------------------+
```
**Ako vidite ovaj izlaz — GPU je dostupan**

**Ako vidite grešku — niste na compute nodu ili niste zatražili GPU resurse u SLURM-u**


### U Python sesiji — provera za PyTorch##
```python
import torch
print(torch.cuda.is_available())      # mora biti True
print(torch.cuda.device_count())      # broj dostupnih GPU-ova
print(torch.cuda.get_device_name(0))  # ime GPU-a, npr. "NVIDIA A100"

# Provera za TensorFlow
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))  # mora biti neprazna lista
```

Ako bilo šta od ovoga vrati False ili praznu listu — stop, ne nastavljaj dalje
problem je ili u SLURM zahtevima ili u Docker konfiguraciji. 

## Sloj 2 pokriva dijagnostiku. ##

**1.2 SLURM** — kako zatražiti GPU

Ovo je najčešće mesto greške broj jedan. Korisnici napišu SLURM skriptu, 
ne navedu GPU, i job se pokrene na CPU nodu.
Minimalna SLURM skripta sa GPU:
```bash
#!/bin/bash
#SBATCH --job-name=moj_gpu_job
#SBATCH --partition=defq              # VAŽNO: mora biti GPU particija
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1                 # KLJUČNA LINIJA — bez ovoga nema GPU-a
#SBATCH --time=02:00:00
#SBATCH --output=izlaz_%j.log

module load CUDA/12.0               # učitaj CUDA modul (naziv zavisi od klastera)
module load Python/3.10

python moj_skript.py
```


Česte varijacije za --gres:
```bash
#SBATCH --gres=gpu:1          # jedan GPU bilo kojeg tipa
#SBATCH --gres=gpu:a100:1     # jedan A100 specifično
#SBATCH --gres=gpu:2          # dva GPU-a (za multi-GPU trening)
```

Kako proveriti da li je SLURM dodelio GPU:

## Unutar pokrenute sesije ili na početku skripte ##
```bash
echo "Dodeljeni GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi
```

## 1.3 Docker — --gpus parametar ##

Greška broj dva je pokretanje Docker kontejnera bez --gpus parametra.

**POGREŠNO — kontejner ne vidi GPU**
```bash
docker run -it pytorch/pytorch:latest python train.py
```

**ISPRAVNO — jedan GPU**
```bash
docker run --gpus 1 -it pytorch/pytorch:latest python train.py
```

**ISPRAVNO — svi dostupni GPU-ovi (najčešće što treba)**
```bash
docker run --gpus all -it pytorch/pytorch:latest python train.py
```

**ISPRAVNO — specifičan GPU po indeksu**
```bash
docker run --gpus '"device=0"' -it pytorch/pytorch:latest python train.py
docker run --gpus '"device=0,1"' -it pytorch/pytorch:latest python train.py  # dva GPU-a
```

Primer kompletne Docker komande za ML trening:
```bash
docker run --gpus all \
  --rm \                                    # obriši kontejner posle završetka
  -v /home/korisnik/podaci:/data \          # mount lokalnih podataka
  -v /home/korisnik/kod:/workspace \        # mount koda
  -w /workspace \                           # working directory
  pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime \
  python train.py --data /data/dataset
```

**1.4 PyTorch — minimalan primer prebacivanja na GPU**
```python
import torch
```
### Uvek na početku koda definišite device ovako — ne hardcode-ujte 'cuda' ###
```markdown
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Koristim: {device}")

# ============================
# TENSORI
# ============================

# Kreiranje tensora direktno na GPU
x = torch.randn(1000, 1000, device=device)

# Prebacivanje postojećeg tensora na GPU
x_cpu = torch.randn(1000, 1000)
x_gpu = x_cpu.to(device)          # .to(device) je preporučeni način
# ili: x_gpu = x_cpu.cuda()       # stariji način, radi ali manje fleksibilno

# ============================
# MODEL
# ============================

model = torch.nn.Linear(784, 10)
model = model.to(device)           # MODEL mora biti na GPU

# ============================
# TRENING LOOP — najčešće mesto greške
# ============================

optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss()

for batch_x, batch_y in dataloader:
    # PODACI moraju biti na istom uređaju kao model
    batch_x = batch_x.to(device)   # ← ovo se najčešće zaboravlja
    batch_y = batch_y.to(device)   # ← i ovo

    optimizer.zero_grad()
    output = model(batch_x)
    loss = criterion(output, batch_y)
    loss.backward()
    optimizer.step()
```
```

Najčešća greška u PyTorch-u je kada je model na GPU-u ali podaci ostanu na CPU-u. Poruka greške tada izgleda ovako:
```bash
RuntimeError: Expected all tensors to be on the same device, 
but found at least two devices, cuda:0 and cpu!
```

Rešenje je uvek batch_x = batch_x.to(device) unutar trening petlje.

**1.5 TensorFlow — minimalan primer**
TensorFlow je u prednosti jer automatski koristi GPU ako ga vidi, 
ali ima svojih zamki:
```python
import tensorflow as tf

# Provera
gpus = tf.config.list_physical_devices('GPU')
print(f"Dostupni GPU-ovi: {gpus}")

# TF automatski stavlja operacije na GPU — ali možete biti eksplicitni
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    c = tf.matmul(a, b)

# Memory growth — VAŽNO na deljenom klasteru
# Bez ovoga TF može da zauzme sav GPU memoriju odjednom
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

Zamka specifična za klastere: TensorFlow po defaultu zauzme svu slobodnu GPU memoriju čak i za mali model. Na deljenom klasteru to blokira sve ostale korisnike. 
Linija 
```python
set_memory_growth(gpu, True)
```
je obavezna i treba je istaći crvenim slovima u dokumentaciji.

### SLOJ 2 — Dijagnostičke procedure ###

Filozofija ovog sloja: korisnik zna da nešto ne radi kako treba, ali ne zna zašto. Cilj je da ga vodimo kroz sistematično sužavanje problema.

**2.1 Univerzalni dijagnostički protokol**

Pre nego što korisnik krene u bilo koji framework-specifičan dijagnostički postupak, ovo je redosled provera:

1. Da li sam na pravom nodu? (gpu vs cpu particija u SLURM-u)
2. Da li je SLURM dodelio GPU? (--gres=gpu:X)
3. Da li sistem vidi GPU? (nvidia-smi)
4. Da li Python/framework vidi GPU? (framework-specifična provera)
5. Da li moj kod KORISTI GPU? (GPU utilization > 0%)

Svaka od ovih tačaka ima svoju proveru:

### Tačka 1 — koji node koristim? ###
```bash
hostname
squeue -u $USER    # pogledaj NODELIST kolonu
```

### Tačka 2 — šta je SLURM dodelio? ###
```bash
echo $CUDA_VISIBLE_DEVICES    # mora biti broj, ne prazan string
scontrol show job $SLURM_JOB_ID | grep GPU
```

### Tačka 3 — nvidia-smi (viđeno u Sloju 1) ###
```bash
nvidia-smi
```

### Tačka 4 — monitoring u realnom vremenu dok job radi ###
```bash
watch -n 1 nvidia-smi    # osvežava svake sekunde
```

Ako **$CUDA_VISIBLE_DEVICES** je prazan — SLURM nije dodelio GPU, proveriti --gres parametar. Ako nvidia-smi pokazuje 0% GPU utilization dok kod radi — framework ne šalje proračune na GPU.

## 2.2 PyTorch dijagnostika ##

Scenario: kod radi, ali sumnjam da ne koristi GPU
```python
import torch

# ---- PROVERA 1: Da li CUDA uopšte postoji u ovoj instalaciji? ----
print(torch.version.cuda)          # treba da bude npr. "12.1", ne None
print(torch.backends.cudnn.enabled)  # treba True

# ---- PROVERA 2: Gde se zapravo nalazi model? ----
model = MojModel()
model.to(device)

# Ovo pokazuje na kom uređaju su parametri modela
for name, param in model.named_parameters():
    print(f"{name}: {param.device}")
# Sve treba da piše cuda:0, ne cpu

# ---- PROVERA 3: Gde su tensori u trening loopu? ----
for batch_x, batch_y in dataloader:
    print(f"batch_x device: {batch_x.device}")  # dodaj ovu liniju privremeno
    print(f"batch_y device: {batch_y.device}")
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    print(f"posle .to(): {batch_x.device}")     # mora biti cuda:0
    break  # samo prvi batch za dijagnostiku

# ---- PROVERA 4: Merenje vremena — CPU vs GPU ----
import time

size = 10000
a = torch.randn(size, size)
b = torch.randn(size, size)

# CPU
start = time.time()
c_cpu = torch.matmul(a, b)
print(f"CPU: {time.time() - start:.3f}s")

# GPU
a_gpu = a.to('cuda')
b_gpu = b.to('cuda')
torch.cuda.synchronize()  # VAŽNO: bez ovoga merenje nije tačno
start = time.time()
c_gpu = torch.matmul(a_gpu, b_gpu)
torch.cuda.synchronize()  # čekaj da GPU završi
print(f"GPU: {time.time() - start:.3f}s")

# Ako su vremena ista — GPU ne radi kako treba
# Ako je GPU brži 10-100x — sve je u redu
```

Scenario: **CUDA out of memory greška**

## Poruka greške: ##
```bash
RuntimeError: CUDA out of memory. Tried to allocate X GiB
```

## Dijagnostika — koliko memorije koristim? ##
```python
print(f"Zauzeto: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Rezervisano: {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

## Česta rešenja: ##

## 1. Smanji batch size (najbrže rešenje) ##
```python
batch_size = 32  # probaj 16 ili 8
```

## 2. Koristi gradient checkpointing za velike modele ##
```python
from torch.utils.checkpoint import checkpoint_sequential
# ovo troši više CPU vremena ali drastično smanjuje GPU memoriju
```

##  3. Koristiti mixed precision (FP16 umesto FP32 — duplo manji memory footprint) ##
```python
scaler = torch.cuda.amp.GradScaler()
with torch.autocast(device_type='cuda', dtype=torch.float16):
    output = model(batch_x)
    loss = criterion(output, batch_y)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 4. Oslobodi memoriju eksplicitno između jobova ##
```python
torch.cuda.empty_cache()
```

## 2.3 TensorFlow dijagnostika ##
```python
import tensorflow as tf

# ---- PROVERA 1: Logging gde TF stavlja operacije ----
tf.debugging.set_log_device_placement(True)
# Ovo štampa svaku operaciju i gde se izvršava — verbose, ali korisno za dijagnostiku

# Primer izlaza:
# Executing op MatMul in device /job:localhost/replica:0/task:0/device:GPU:0
# Ako piše CPU umesto GPU — operacija nije na GPU-u

# ---- PROVERA 2: Eksplicitna alokacija ----
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"TF vidi {len(gpus)} GPU-ova")
    for gpu in gpus:
        print(f"  {gpu}")
else:
    print("TF NE VIDI nijedan GPU")
    print("Proverite: nvidia-smi, CUDA_VISIBLE_DEVICES, instalaciju TF-GPU verzije")

# ---- PROVERA 3: Da li je instalirana GPU verzija TF-a? ----
print(tf.__version__)
# Ovo samo po sebi ne govori mnogo, ali:
python -c "import tensorflow as tf; print(tf.test.is_built_with_cuda())"
# mora biti True

# ---- ČESTA ZAMKA: instalirana je CPU verzija TF-a ----
# pip install tensorflow           → može biti CPU verzija
# pip install tensorflow[and-cuda] → eksplicitno GPU verzija (TF 2.12+)
# conda install tensorflow-gpu     → GPU verzija kroz conda


Scenario: TF zauzima svu GPU memoriju

# Simptom: drugi korisnici na istom nodu dobijaju OOM greške
# ili vaš job odmah pada sa OOM iako model nije velik

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    # Opcija 1: Memory growth — zauzima onoliko koliko treba
    tf.config.experimental.set_memory_growth(gpu, True)
    
    # Opcija 2: Tvrdo ograničenje u MB
    tf.config.set_logical_device_configuration(
        gpu,
        [tf.config.LogicalDeviceConfiguration(memory_limit=8192)]  # 8GB limit
    )

# NAPOMENA: ovo mora biti pozvano PRE bilo kakvih GPU operacija
# Ako dobijete grešku "Physical devices cannot be modified after being initialized"
# — prekasno ste pozvali ovu funkciju, mora ići na sam početak skripte
```

## 2.4 CuPy — NumPy kod koji ne koristi GPU ##

Ovo je slučaj koji korisnici često previde. Imaju numerički kod u NumPy-u i misle da automatski koristi GPU. Ne koristi.
```python
import numpy as np
import cupy as cp
import time

size = 5000

# ---- NumPy — CPU ----
a_np = np.random.randn(size, size).astype(np.float32)
b_np = np.random.randn(size, size).astype(np.float32)

start = time.time()
c_np = np.dot(a_np, b_np)
print(f"NumPy (CPU): {time.time() - start:.3f}s")

# ---- CuPy — GPU, API identičan NumPy-u ----
a_cp = cp.asarray(a_np)  # prebaci na GPU
b_cp = cp.asarray(b_np)

cp.cuda.Stream.null.synchronize()
start = time.time()
c_cp = cp.dot(a_cp, b_cp)
cp.cuda.Stream.null.synchronize()
print(f"CuPy  (GPU): {time.time() - start:.3f}s")

# Rezultat: GPU je tipično 10-50x brži za ovakve operacije

# Prebacivanje rezultata nazad na CPU ako treba
c_back_on_cpu = cp.asnumpy(c_cp)
```

Migracija postojećeg NumPy koda — za jednostavne slučajeve može biti gotovo trivijalna:

### STARO ###
```python
import numpy as np
xp = np
```

### NOVO — jedna promena, ostatak koda identičan ###
```python
import cupy as cp
xp = cp

# Zatim koristite xp umesto np u kodu:
a = xp.random.randn(1000, 1000)
b = xp.random.randn(1000, 1000)
c = xp.dot(a, b)  # ovo sada ide na GPU
```

Naravno, ovo radi samo ako koristite standardne NumPy operacije. Ako koristite scipy, sklearn ili custom C ekstenzije — situacija je komplikovanija.

## 2.5 Docker dijagnostika — kompletno stablo odlučivanja ##

### KORAK 1: Da li host vidi GPU? ###
```bash
nvidia-smi
```
**Greška → problem je na nivou drivera, kontaktirajte admin**

### KORAK 2: Pokrenite test kontejner ###
```bash
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi
# Ako ovo radi ali vaš kontejner ne — problem je u vašoj Docker slici
```

### KORAK 3: Da li vaša Docker slika ima CUDA? ###
```dockerfile
# POGREŠNA baza — nema CUDA
FROM python:3.10-slim

# ISPRAVNA baza — ima CUDA runtime
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
# ili koristite zvanične slike sa CUDA:
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
```

### KORAK 4: Provera unutar kontejnera ###
```bash
docker run --gpus all -it pytorch/pytorch:latest bash
# Unutar kontejnera:
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

Česta greška sa Docker Compose:

## docker-compose.yml ##
```yaml
# POGREŠNO — nema GPU pristup
services:
  training:
    image: pytorch/pytorch:latest

# ISPRAVNO
services:
  training:
    image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all              # ili count: 1 za jedan GPU
              capabilities: [gpu]
```

## 2.6 Dijagnostika u realnom vremenu — monitoring tokom job-a ##

Ovo je korisno naučiti jer omogućava korisniku da prati da li job zapravo koristi GPU dok radi:

**Terminal 1 — pokrenite job**
```bash
sbatch moj_job.sh
```

**Terminal 2 — monitoring (pokrenite na istom compute nodu)**
```bash
ssh ime-compute-noda    # ili koristite srun --pty bash u istom jobu
watch -n 2 nvidia-smi  # osvežava na svake 2 sekunde
```

## Šta gledati u nvidia-smi izlazu: ##

- GPU-Util: trebalo bi da bude > 50% tokom aktivnog treninga
- Memory-Usage: trebalo bi da raste posle učitavanja modela
- Power Usage: viši = GPU stvarno radi nešto

## Detaljniji monitoring ##
```bash
nvidia-smi dmon -s u    # utilization sampling svake sekunde
nvidia-smi dmon -s m    # memory sampling

# Za Python profiling — koliko vremena prolazi na GPU vs CPU
# pip install torch-tb-profiler
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with record_function("model_inference"):
        output = model(input_data)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```
