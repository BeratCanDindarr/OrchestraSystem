using System;
using UnityEngine;

namespace ForestBuildingSystem.Data
{
    [Serializable]
    public struct ResourceRequirement
    {
        public ResourceSO resource;
        public int amount;
    }
}
