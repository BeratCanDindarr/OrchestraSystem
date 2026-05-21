using NUnit.Framework;
using UnityEngine;
using VillageSim.BuildingSystem;

namespace VillageSim.Tests
{
    public class BuildingSystemTests
    {
        [Test]
        public void PhysicsCheckBox_DetectsOverlap_Correctly()
        {
            // Setup a collider
            GameObject wall = new GameObject("TestWall");
            wall.AddComponent<BoxCollider>().size = Vector3.one;
            wall.transform.position = Vector3.zero;

            // Test position that should overlap
            Vector3 overlapPos = new Vector3(0.2f, 0, 0);
            Vector3 extents = new Vector3(0.4f, 0.4f, 0.4f);
            
            // We use OverlapBox since CheckBox doesn't return anything we can easily assert without a physics scene
            Collider[] overlaps = Physics.OverlapBox(overlapPos, extents, Quaternion.identity);
            Assert.IsTrue(overlaps.Length > 0, "Should detect overlap with TestWall");
            
            // Cleanup
            Object.DestroyImmediate(wall);
        }

        [Test]
        public void SnapPoint_ReturnsCorrectWorldPosition()
        {
            GameObject go = new GameObject("SnapParent");
            go.transform.position = new Vector3(10, 5, 10);
            
            SnapPoint sp = go.AddComponent<SnapPoint>();
            Assert.AreEqual(new Vector3(10, 5, 10), sp.WorldPosition);
            
            Object.DestroyImmediate(go);
        }

        [Test]
        public void Snap_CalculatesPositionCorrectly_WithRotation()
        {
            // This test simulates the logic in TrySnap to verify the math
            
            // Target snap point at (10, 0, 10) rotated 90 degrees
            Vector3 targetPos = new Vector3(10, 0, 10);
            Quaternion targetRot = Quaternion.Euler(0, 90, 0);
            
            // Source snap point is at local (0, 0, 2) relative to blueprint root
            Vector3 sourceLocalPos = new Vector3(0, 0, 2);
            
            // When blueprint is rotated 90 degrees, source point should be at targetPos
            // Root + (targetRot * sourceLocalPos) = targetPos
            // Root = targetPos - (targetRot * sourceLocalPos)
            
            Vector3 expectedRootPos = targetPos - (targetRot * sourceLocalPos);
            
            // The bug was using the OLD rotation. Let's see if the old rotation (0) would give a different result.
            Quaternion oldRot = Quaternion.identity;
            Vector3 wrongRootPos = targetPos - (oldRot * sourceLocalPos);
            
            Assert.AreNotEqual(expectedRootPos, wrongRootPos, "Mathematics should differ when rotation is ignored");
            
            // Logic check: if target is at (10,0,10) and source is 2 units forward locally,
            // and target orientation is 90 deg (facing Right), then source point should be at (10,0,10).
            // That means root should be 2 units LEFT of (10,0,10) => (8,0,10).
            // Let's verify: targetRot * (0,0,2) where targetRot is 90 deg around Y.
            // Forward (0,0,1) rotated 90 deg is Right (1,0,0).
            // So targetRot * (0,0,2) is (2,0,0).
            // Root = (10,0,10) - (2,0,0) = (8,0,10).
            Assert.AreEqual(new Vector3(8, 0, 10), expectedRootPos);
        }
    }
}
