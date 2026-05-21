using UnityEngine;
using System.Collections.Generic;

namespace VillageSim.BuildingSystem.Features
{
    /// <summary>
    /// Handles the placement of blueprints with snapping and collision validation.
    /// </summary>
    public class BlueprintPlacer : MonoBehaviour
    {
        [Header("Placement Settings")]
        [SerializeField] private LayerMask _groundLayer;
        [SerializeField] private LayerMask _snapLayer;
        [SerializeField] private float _maxDistance = 15f;
        [SerializeField] private float _snapRadius = 2.0f;
        [SerializeField] private Vector3 _boxCheckHalfExtents = new Vector3(0.45f, 0.45f, 0.45f);

        [Header("Input Keys")]
        [SerializeField] private KeyCode _toggleSnapKey = KeyCode.S;
        [SerializeField] private KeyCode _rotateKey = KeyCode.R;

        private GameObject _activeBlueprint;
        private List<SnapPoint> _blueprintSnapPoints = new List<SnapPoint>();
        private bool _isSnappingEnabled = true;
        private float _currentRotation = 0f;

        public void SetBlueprint(GameObject prefab)
        {
            if (_activeBlueprint != null) Destroy(_activeBlueprint);
            
            _activeBlueprint = Instantiate(prefab);
            _blueprintSnapPoints.Clear();
            _blueprintSnapPoints.AddRange(_activeBlueprint.GetComponentsInChildren<SnapPoint>());
            
            SetLayerRecursive(_activeBlueprint, LayerMask.NameToLayer("Ignore Raycast"));
        }

        private void Update()
        {
            if (_activeBlueprint == null) return;

            if (Input.GetKeyDown(_toggleSnapKey)) _isSnappingEnabled = !_isSnappingEnabled;
            if (Input.GetKeyDown(_rotateKey)) _currentRotation += 90f;

            HandlePlacement();

            if (Input.GetMouseButtonDown(0) && CanPlace())
            {
                Place();
            }
        }

        private void HandlePlacement()
        {
            Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            if (Physics.Raycast(ray, out RaycastHit hit, _maxDistance, _groundLayer))
            {
                Vector3 targetPos = hit.point;
                Quaternion targetRot = Quaternion.Euler(0, _currentRotation, 0);

                if (_isSnappingEnabled)
                {
                    TrySnap(ref targetPos, ref targetRot);
                }

                _activeBlueprint.transform.position = targetPos;
                _activeBlueprint.transform.rotation = targetRot;
            }
        }

        private void TrySnap(ref Vector3 pos, ref Quaternion rot)
        {
            Collider[] colliders = Physics.OverlapSphere(pos, _snapRadius, _snapLayer);
            SnapPoint bestTarget = null;
            SnapPoint bestSource = null;
            float minDistance = float.MaxValue;

            foreach (var col in colliders)
            {
                SnapPoint[] targetPoints = col.GetComponentsInChildren<SnapPoint>();
                foreach (var targetPoint in targetPoints)
                {
                    foreach (var sourcePoint in _blueprintSnapPoints)
                    {
                        if (targetPoint.SnapType != sourcePoint.SnapType) continue;

                        float dist = Vector3.Distance(pos, targetPoint.WorldPosition);
                        if (dist < minDistance)
                        {
                            minDistance = dist;
                            bestTarget = targetPoint;
                            bestSource = sourcePoint;
                        }
                    }
                }
            }

            if (bestTarget != null)
            {
                // Align so source matches target position and rotation
                rot = bestTarget.transform.rotation;
                
                // Calculate local position of the source snap point relative to the blueprint root
                Vector3 localSourcePos = _activeBlueprint.transform.InverseTransformPoint(bestSource.WorldPosition);
                
                // Calculate the world offset at the NEW rotation
                Vector3 sourceOffset = rot * localSourcePos;
                
                // Set the root position so the source point aligns with the target point
                pos = bestTarget.WorldPosition - sourceOffset;
            }
        }

        private bool CanPlace()
        {
            Collider[] overlaps = Physics.OverlapBox(_activeBlueprint.transform.position, _boxCheckHalfExtents, _activeBlueprint.transform.rotation, ~LayerMask.GetMask("Ignore Raycast"));
            return overlaps.Length == 0;
        }

        private void Place()
        {
            SetLayerRecursive(_activeBlueprint, LayerMask.NameToLayer("Default"));
            _activeBlueprint.GetComponent<BlueprintStructure>()?.Initialize(null); // Or pass recipe
            _activeBlueprint = null;
            _currentRotation = 0f;
        }

        private void SetLayerRecursive(GameObject obj, int layer)
        {
            obj.layer = layer;
            foreach (Transform child in obj.transform) SetLayerRecursive(child.gameObject, layer);
        }
    }
}
