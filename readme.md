# [for ML study]
## hyperparameter description
1. learning_rate: 학습률은 모델의 학습 속도와 성능을 크게 좌우합니다. 0.01에서 0.1 사이의 값을 주로 사용하며, 작을수록 더 많은 트리(n_estimators)가 필요합니다.
2. n_estimators: 부스팅 과정에서 생성할 트리의 개수를 결정합니다. learning_rate가 낮을수록 트리 개수를 늘리는 것이 좋습니다.  
※작은 learning_rate, 큰 n_estimators가 더 자주 사용됩니다. 이 방식은 모델이 더 천천히, 그러나 안정적으로 학습하도록 하여 과적합을 방지합니다.  
예를 들어, learning_rate를 0.01로 설정했다면, n_estimators를 `500~1000` 이상으로 설정하는 것이 일반적입니다. 반대로, learning_rate를 0.1로 설정한다면, n_estimators를 `100~300` 정도로 설정할 수 있습니다.
4. num_leaves: 트리의 복잡도를 결정하는 파라미터입니다. 더 많은 리프가 있을수록 트리가 복잡해져 성능이 좋아질 수 있지만, 과적합의 위험도 증가합니다. 데이터 크기나 특성에 맞게 적절히 조정해야 합니다.
5. min_data_in_leaf: 리프 노드에 있는 최소 데이터 수로, 과적합을 방지하는 데 중요한 역할을 합니다. 작은 값은 모델이 세부 사항을 더 학습하게 만들고, 큰 값은 모델이 단순화됩니다.
6. max_depth: 트리의 최대 깊이를 제한하여 과적합을 방지할 수 있습니다. 기본값은 -1로 제한이 없지만, 트리의 복잡도를 제한하고 싶다면 적절한 값을 설정하세요.  
※일반적으로, num_leaves는 2^(max_depth)보다 작게 설정하는 것이 좋습니다. 예를 들어, max_depth=6이라면 num_leaves는 64 이하로 설정합니다.

[default value(lightgbm)]  
learning_rate: 0.1  
n_estimators: 100  
num_leaves: 31  
min_data_in_leaf: 20  
max_depth: -1  
