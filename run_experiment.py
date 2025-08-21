import os
import time
from generate_shapes import generate_data
from train_model import train_model

NUM_IMAGES_TO_GENERATE = 15_000
IMAGE_SIZE = 28
NUM_EPOCHS = 40
BATCH_SIZE = 64
LEARNING_RATE = 0.001

if __name__ == "__main__":

    exp_start = time.time()

    dataset_folder_path = generate_data(
        num_images = NUM_IMAGES_TO_GENERATE,
        img_size = IMAGE_SIZE
    )

    final_acc = train_model(
        dataset_path = dataset_folder_path,
        batch_size = BATCH_SIZE,
        epochs = NUM_EPOCHS,
        learning_rate = LEARNING_RATE,
        num_images = NUM_IMAGES_TO_GENERATE
    )

    exp_end = time.time()

    tempo_exe = exp_start - exp_start
    min = tempo_exe//60
    seg = tempo_exe%60

    duracao = f"{int(min)} min(s) e {seg:.2f} seg(s)"


    summary_file_path = os.path.join(dataset_folder_path, 'parametros_do_modelo.txt')
    summary_content = f"""
Relatório Final do modelo:

Número de imagens = {NUM_IMAGES_TO_GENERATE}
Tamanho = {IMAGE_SIZE}x{IMAGE_SIZE}
Épocas = {NUM_EPOCHS}
Batch = {BATCH_SIZE}
Learning Rate = {LEARNING_RATE}

Acurácia final do modelo = {final_acc:.2f}%
Duração = {duracao}
"""
    with open(summary_file_path, 'w', encoding='utf-8') as f:
        f.write(summary_content)