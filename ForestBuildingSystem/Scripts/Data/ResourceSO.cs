using UnityEngine;
using VillageSim.BuildingSystem.Core;

namespace ForestBuildingSystem.Data
{
    [CreateAssetMenu(fileName = "NewResource", menuName = "Forest System/Resource")]
    public class ResourceSO : ScriptableObject, IResourceIdentity
    {
        public string resourceName;
        public Sprite icon;

        public string ResourceId => resourceName; // Simplified for this implementation
        public string Name => resourceName;
    }
}
