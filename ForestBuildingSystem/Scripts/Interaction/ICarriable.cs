using UnityEngine;
using ForestBuildingSystem.Data;

namespace ForestBuildingSystem.Interaction
{
    public interface ICarriable
    {
        ResourceSO ResourceType { get; }
        Transform Transform { get; }
        void OnPickup(Transform carrier);
        void OnTransferToSite();
        void OnDrop(Vector3 force);
    }
}
