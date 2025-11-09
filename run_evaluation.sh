#!/bin/bash
echo "Iniciando loop de AVALIAÇÃO de modelos..."

# --- Configurações ---
# Parâmetros para iterar
IMG_SIZES=(32 64 128 256)
LSTM_UNITS=(256 512 1024 2048 4096)

# Caminho para o NOVO script de avaliação
# (Assumindo que está no mesmo diretório do script de treino)
PYTHON_SCRIPT_PATH="/home/vitorlisboa/tcc/lstm/experimentos_videos_alfabeto/avaliacao.py"

# Argumentos necessários para o avaliacao.py
DATA_DIR="/home/vitorlisboa/datasets/videos_alfabeto_cropped/breno"
# Diretório onde as pastas 'checkpoints_...' estão salvas
CHECKPOINT_BASE_DIR="/home/vitorlisboa/tcc/"
SEQ_LENGTH=32
BATCH_SIZE=2 # Pode aumentar o batch size na avaliação se a memória permitir
SEED=42

# Loop pelos parâmetros
for units in "${LSTM_UNITS[@]}"; do
    for size in "${IMG_SIZES[@]}"; do

        # Nome do arquivo de *resultado da avaliação* para verificar se já existe
        RESULTS_FILE="${CHECKPOINT_BASE_DIR}/evaluation_results_${size}x${size}_${units}_report.txt"
        
        # Caminho do modelo treinado que servirá de *entrada*
        MODEL_FILE_PATH="${CHECKPOINT_BASE_DIR}/checkpoints_${size}x${size}_${units}/best_model.h5"


        if [ -f "$RESULTS_FILE" ]; then
            echo "---"
            echo "PULANDO: $RESULTS_FILE já existe. (Size: $size, Units: $units)"
            echo "---"
        
        elif [ ! -f "$MODEL_FILE_PATH" ]; then
            # Se o modelo treinado não existir, não há o que avaliar
            echo "---"
            echo "ERRO: Modelo $MODEL_FILE_PATH não encontrado. Pulando avaliação. (Size: $size, Units: $units)"
            echo "---"
        
        else
            echo "========================================================================"
            echo "INICIANDO AVALIAÇÃO: Size: $size, Units: $units"
            echo "========================================================================"

            # Chama o script Python de AVALIAÇÃO com todos os argumentos
            python3 "$PYTHON_SCRIPT_PATH" \
                --img_size "$size" \
                --lstm_units "$units" \
                #--data_dir "$DATA_DIR" \
                #--checkpoint_dir "$CHECKPOINT_BASE_DIR" \
                #--sequence_length "$SEQ_LENGTH" \
                #--batch_size "$BATCH_SIZE" \
                #--seed "$SEED"
        fi
    done
done

echo "Todas as avaliações concluídas."
