using System;
using System.Collections.Generic;
using System.Linq;

namespace VillageSim.BuildingSystem.Core
{
    /// <summary>
    /// Pure C# model for tracking building progress and resource requirements.
    /// </summary>
    public class BlueprintModel
    {
        private readonly Dictionary<string, int> _requiredResources = new Dictionary<string, int>();
        private readonly Dictionary<string, int> _currentResources = new Dictionary<string, int>();

        public event Action OnChanged;
        public event Action OnCompleted;

        public bool IsComplete => _requiredResources.Count > 0 && _requiredResources.All(r => GetCurrentAmount(r.Key) >= r.Value);

        public void Initialize(IEnumerable<(IResourceIdentity resource, int amount)> requirements)
        {
            _requiredResources.Clear();
            _currentResources.Clear();

            foreach (var req in requirements)
            {
                if (req.amount <= 0) continue;
                _requiredResources[req.resource.ResourceId] = req.amount;
            }
            OnChanged?.Invoke();
        }

        public bool AddResource(IResourceIdentity resource, int amount)
        {
            if (amount <= 0 || !CanAccept(resource)) return false;

            string id = resource.ResourceId;
            int required = _requiredResources[id];
            int current = GetCurrentAmount(id);
            int accepted = Math.Min(amount, required - current);

            if (accepted <= 0) return false;

            if (!_currentResources.ContainsKey(id)) _currentResources[id] = 0;
            _currentResources[id] += accepted;

            OnChanged?.Invoke();
            if (IsComplete) OnCompleted?.Invoke();

            return true;
        }

        public bool CanAccept(IResourceIdentity resource)
        {
            if (resource == null) return false;
            string id = resource.ResourceId;
            return _requiredResources.ContainsKey(id) && GetCurrentAmount(id) < _requiredResources[id];
        }

        public int GetCurrentAmount(string resourceId)
        {
            return _currentResources.TryGetValue(resourceId, out int amount) ? amount : 0;
        }

        public int GetRequiredAmount(string resourceId)
        {
            return _requiredResources.TryGetValue(resourceId, out int amount) ? amount : 0;
        }

        public float GetProgress()
        {
            if (_requiredResources.Count == 0) return 0f;
            float totalRequired = _requiredResources.Values.Sum();
            float totalCurrent = _requiredResources.Keys.Sum(id => Math.Min(GetCurrentAmount(id), _requiredResources[id]));
            return totalCurrent / totalRequired;
        }
    }
}
