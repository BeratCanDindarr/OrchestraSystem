using UnityEngine;
using System.Collections.Generic;
using System.Linq;
using ForestBuildingSystem.Data;
using VillageSim.BuildingSystem.Core;
using ForestBuildingSystem.Interaction;

namespace VillageSim.BuildingSystem.Features
{
    /// <summary>
    /// View/Controller for the building blueprint. Handles Unity-specific logic
    /// and delegates state to BlueprintModel.
    /// </summary>
    public class BlueprintStructure : MonoBehaviour
    {
        [Header("Recipe")]
        [SerializeField] private BuildingRecipeSO _recipe;
        
        [Header("Construction Visuals")]
        [SerializeField] private List<GameObject> _constructionStages;
        [SerializeField] private Transform _resourceLerpTarget;
        [SerializeField] private float _resourceLerpDuration = 0.2f;

        [Header("Magnet Interaction")]
        [SerializeField] private Collider _magnetTrigger;
        [SerializeField] private LayerMask _playerLayer = ~0;

        private BlueprintModel _model = new BlueprintModel();
        private readonly Collider[] _overlapResults = new Collider[8];
        private bool _isReceivingResource;

        public BuildingRecipeSO Recipe => _recipe;
        public bool IsComplete => _model.IsComplete;

        private void Awake()
        {
            _model.OnChanged += UpdateVisuals;
            _model.OnCompleted += FinalizeBuilding;

            if (_resourceLerpTarget == null) _resourceLerpTarget = transform;
            if (_magnetTrigger != null) _magnetTrigger.isTrigger = true;

            if (_recipe != null) Initialize(_recipe);
        }

        public void Initialize(BuildingRecipeSO recipe)
        {
            _recipe = recipe;
            var requirements = _recipe.requirements.Select(r => ((IResourceIdentity)r.resource, r.amount));
            _model.Initialize(requirements);
            UpdateVisuals();
        }

        private void Update()
        {
            if (_isReceivingResource || _magnetTrigger == null || _recipe == null || IsComplete) return;
            TryReceiveNearbyResource();
        }

        public bool AddResource(ResourceSO resource, int amount)
        {
            return _model.AddResource(resource, amount);
        }

        public bool CanAccept(ResourceSO resource)
        {
            return _model.CanAccept(resource);
        }

        private void TryReceiveNearbyResource()
        {
            Bounds bounds = _magnetTrigger.bounds;
            int hitCount = Physics.OverlapBoxNonAlloc(
                bounds.center,
                bounds.extents,
                _overlapResults,
                Quaternion.identity,
                _playerLayer,
                QueryTriggerInteraction.Collide);

            for (int i = 0; i < hitCount; i++)
            {
                PlayerCarryingSystem carryingSystem = _overlapResults[i].GetComponentInParent<PlayerCarryingSystem>();
                if (carryingSystem == null || !carryingSystem.IsCarrying) continue;

                ResourceSO resource = carryingSystem.CurrentItem.ResourceType;
                if (!CanAccept(resource)) continue;

                if (!carryingSystem.TryReleaseCarriedItem(out ICarriable carriedItem)) continue;

                carriedItem.OnTransferToSite();
                StartCoroutine(LerpResourceToSite(carriedItem));
                _isReceivingResource = true;
                return;
            }
        }

        private System.Collections.IEnumerator LerpResourceToSite(ICarriable carriedItem)
        {
            Transform carriedTransform = carriedItem.Transform;
            Vector3 startPosition = carriedTransform.position;
            Quaternion startRotation = carriedTransform.rotation;
            float elapsed = 0f;

            while (elapsed < _resourceLerpDuration)
            {
                elapsed += Time.deltaTime;
                float t = _resourceLerpDuration <= 0f ? 1f : Mathf.Clamp01(elapsed / _resourceLerpDuration);
                carriedTransform.position = Vector3.Lerp(startPosition, _resourceLerpTarget.position, t);
                carriedTransform.rotation = Quaternion.Slerp(startRotation, _resourceLerpTarget.rotation, t);
                yield return null;
            }

            _model.AddResource(carriedItem.ResourceType, 1);
            Destroy(carriedTransform.gameObject);
            _isReceivingResource = false;
        }

        private void UpdateVisuals()
        {
            if (_constructionStages == null || _constructionStages.Count == 0) return;

            float progress = _model.GetProgress();
            int stageIndex = Mathf.Clamp(Mathf.FloorToInt(progress * _constructionStages.Count), 0, _constructionStages.Count - 1);

            for (int i = 0; i < _constructionStages.Count; i++)
            {
                _constructionStages[i].SetActive(i <= stageIndex);
            }
        }

        private void FinalizeBuilding()
        {
            if (_recipe.buildingPrefab != null)
            {
                Instantiate(_recipe.buildingPrefab, transform.position, transform.rotation);
            }
            Destroy(gameObject);
        }

        private void OnDestroy()
        {
            _model.OnChanged -= UpdateVisuals;
            _model.OnCompleted -= FinalizeBuilding;
        }
    }
}
