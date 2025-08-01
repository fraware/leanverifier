import FormalVerifML.base.advanced_models
namespace FormalVerifML
-- Auto-generated Transformer definition for sampleTransformer

def sampleTransformer : Transformer :=
  { dModel := 64,
    numHeads := 4,
    numLayers := 2,
    vocabSize := 1000,
    maxSeqLen := 128,
    tokenEmbeddings := [
    #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    #[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  ],
    positionalEmbeddings := [
    #[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
    #[0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
  ],
    attentionHeads := [
    [
      AttentionHead.mk [
        #[0.1, 0.2],
        #[0.3, 0.4]
      ] [
        #[0.5, 0.6],
        #[0.7, 0.8]
      ] [
        #[0.9, 1.0],
        #[1.1, 1.2]
      ] [
        #[0.1, 0.2],
        #[0.3, 0.4]
      ],
      AttentionHead.mk [
        #[0.2, 0.3],
        #[0.4, 0.5]
      ] [
        #[0.6, 0.7],
        #[0.8, 0.9]
      ] [
        #[1.0, 1.1],
        #[1.2, 1.3]
      ] [
        #[0.2, 0.3],
        #[0.4, 0.5]
      ]
    ],
    [
      AttentionHead.mk [
        #[0.3, 0.4],
        #[0.5, 0.6]
      ] [
        #[0.7, 0.8],
        #[0.9, 1.0]
      ] [
        #[1.1, 1.2],
        #[1.3, 1.4]
      ] [
        #[0.3, 0.4],
        #[0.5, 0.6]
      ],
      AttentionHead.mk [
        #[0.4, 0.5],
        #[0.6, 0.7]
      ] [
        #[0.8, 0.9],
        #[1.0, 1.1]
      ] [
        #[1.2, 1.3],
        #[1.4, 1.5]
      ] [
        #[0.4, 0.5],
        #[0.6, 0.7]
      ]
    ]
  ],
    layerNorms1 := [
    (#[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], #[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    (#[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], #[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
  ],
    layerNorms2 := [
    (#[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], #[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    (#[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], #[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
  ],
    ffWeights1 := [
    ([
      #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
      #[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ], #[0.1, 0.2]),
    ([
      #[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
      #[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    ], #[0.3, 0.4])
  ],
    ffWeights2 := [
    ([
      #[0.1, 0.2],
      #[0.3, 0.4],
      #[0.5, 0.6],
      #[0.7, 0.8],
      #[0.9, 1.0],
      #[1.1, 1.2],
      #[1.3, 1.4],
      #[1.5, 1.6]
    ], #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
    ([
      #[0.2, 0.3],
      #[0.4, 0.5],
      #[0.6, 0.7],
      #[0.8, 0.9],
      #[1.0, 1.1],
      #[1.2, 1.3],
      #[1.4, 1.5],
      #[1.6, 1.7]
    ], #[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
  ],
    outputProjection := ([
    #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
  ], #[0.1]) }
end FormalVerifML