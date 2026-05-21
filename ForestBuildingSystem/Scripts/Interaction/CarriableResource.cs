using UnityEngine;
using ForestBuildingSystem.Data;

namespace ForestBuildingSystem.Interaction
{
    [RequireComponent(typeof(Rigidbody))]
    public class CarriableResource : MonoBehaviour, ICarriable
    {
        [SerializeField] private ResourceSO resourceType;
        
        private Rigidbody rb;
        private Collider col;

        public ResourceSO ResourceType => resourceType;
        public Transform Transform => transform;

        private void Awake()
        {
            rb = GetComponent<Rigidbody>();
            col = GetComponent<Collider>();
        }

        public void OnPickup(Transform carrier)
        {
            rb.linearVelocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
            rb.isKinematic = true;
            col.enabled = false;
            transform.SetParent(carrier);
            transform.localPosition = Vector3.zero;
            transform.localRotation = Quaternion.identity;
        }

        public void OnTransferToSite()
        {
            transform.SetParent(null, true);
            rb.linearVelocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
            rb.isKinematic = true;
            col.enabled = false;
        }

        public void OnDrop(Vector3 force)
        {
            transform.SetParent(null, true);
            rb.isKinematic = false;
            col.enabled = true;
            rb.linearVelocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
            rb.AddForce(force, ForceMode.Impulse);
        }
    }
}
