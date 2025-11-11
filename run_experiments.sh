#!/bin/bash
echo "Iniciando loop de experimentos..."

# Define os parâmetros
IMG_SIZES=(32 64 128 256)
LSTM_UNITS=(256 512 1024 2048 4096)

# Caminho para seu script Python
PYTHON_SCRIPT_PATH="/home/vitorlisboa/tcc/treinamentos/lstm.py"

# Loop pelos parâmetros
for units in "${LSTM_UNITS[@]}"; do
    for size in "${IMG_SIZES[@]}"; do
        
        # Nome do arquivo de resultado para verificar se já existe
        RESULTS_FILE="best_model_results_${size}x${size}_${units}.txt"
        
        if [ -f "$RESULTS_FILE" ]; then
            echo "---"
            echo "PULANDO: $RESULTS_FILE já existe. (Size: $size, Units: $units)"
            echo "---"
        else
            echo "========================================================================"
            echo "INICIANDO: Size: $size, Units: $units (Usando ambas as GPUs)"
            echo "========================================================================"
            
            python3 "$PYTHON_SCRIPT_PATH" --img_size "$size" --lstm_units "$units"
        fi
    done
done

echo "Todos os experimentos concluídos."
