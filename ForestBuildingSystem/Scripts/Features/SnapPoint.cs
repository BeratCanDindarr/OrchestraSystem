using UnityEngine;

namespace VillageSim.BuildingSystem
{
    /// <summary>
    /// Defines a point where other structures can snap to.
    /// </summary>
    public class SnapPoint : MonoBehaviour
    {
        [SerializeField] private string _snapType = "General";

        public Vector3 WorldPosition => transform.position;
        public string SnapType => _snapType;

        private void OnDrawGizmos()
        {
            Gizmos.color = Color.cyan;
            Gizmos.DrawWireSphere(transform.position, 0.15f);
            Gizmos.DrawRay(transform.position, transform.forward * 0.3f);
        }
    }
}
