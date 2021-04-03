# neural-fluids
learning to imitate fluid simulations

    python corpus_builder.py
    convert placeholder_*.png placeholder.gif
    python train.py corpus.npy
    python evaluate.py model.mdl corpus.npy
    convert compare_0*.png compare.gif
    convert compare.gif -resize 200%x200% compare2.gif
