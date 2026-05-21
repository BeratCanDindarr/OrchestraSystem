using UnityEngine;
using System.Collections.Generic;

namespace ForestBuildingSystem.Data
{
    [CreateAssetMenu(fileName = "NewBuildingRecipe", menuName = "Forest System/Building Recipe")]
    public class BuildingRecipeSO : ScriptableObject
    {
        public string buildingName;
        public List<ResourceRequirement> requirements;
        public GameObject buildingPrefab;
    }
}
