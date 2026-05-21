namespace VillageSim.BuildingSystem.Core
{
    /// <summary>
    /// Pure C# interface for identifying resources without UnityEngine dependencies.
    /// </summary>
    public interface IResourceIdentity
    {
        string ResourceId { get; }
        string Name { get; }
    }
}
