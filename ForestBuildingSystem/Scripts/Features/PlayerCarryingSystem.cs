using UnityEngine;
using ForestBuildingSystem.Interaction;

namespace VillageSim.BuildingSystem.Features
{
    public class PlayerCarryingSystem : MonoBehaviour
    {
        [Header("Settings")]
        [SerializeField] private float pickupRange = 3f;
        [SerializeField] private LayerMask pickupLayer;
        [SerializeField] private LayerMask obstacleLayer;
        [SerializeField] private Transform carryPoint;
        [SerializeField] private float dropForce = 5f;

        private ICarriable currentCarriedItem;

        public bool IsCarrying => currentCarriedItem != null;
        public ICarriable CurrentItem => currentCarriedItem;

        // For Testing
        public void Initialize(LayerMask pickup, LayerMask obstacle, Transform point)
        {
            pickupLayer = pickup;
            obstacleLayer = obstacle;
            carryPoint = point;
        }

        public void TryPickup()
        {
            if (IsCarrying)
            {
                Debug.Log("Already carrying an item.");
                return;
            }

            Ray ray = new Ray(transform.position, transform.forward);
            if (Physics.Raycast(ray, out RaycastHit hit, pickupRange, pickupLayer | obstacleLayer))
            {
                // Edge Case: Check if we hit an obstacle first (picking through walls)
                if (((1 << hit.collider.gameObject.layer) & obstacleLayer) != 0)
                {
                    Debug.Log("Pickup blocked by obstacle.");
                    return;
                }

                if (hit.collider.TryGetComponent(out ICarriable carriable))
                {
                    Pickup(carriable);
                }
            }
        }

        public void Drop()
        {
            if (!IsCarrying) return;

            ICarriable itemToDrop = ReleaseCarriedItem();
            itemToDrop.OnDrop(transform.forward * dropForce);
        }

        public bool TryReleaseCarriedItem(out ICarriable item)
        {
            item = null;

            if (!IsCarrying)
            {
                return false;
            }

            item = ReleaseCarriedItem();
            return true;
        }

        private void Pickup(ICarriable carriable)
        {
            currentCarriedItem = carriable;
            carriable.OnPickup(carryPoint);
        }

        private ICarriable ReleaseCarriedItem()
        {
            ICarriable itemToRelease = currentCarriedItem;
            currentCarriedItem = null;
            return itemToRelease;
        }
    }
}
