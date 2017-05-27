import os
from tempfile import NamedTemporaryFile
from unittest import TestCase, main
from polynondom import AssignmentDomain, CubeDomain, Objectives, PolyNondom
from polynondom import InfeasibleBoxError


class DomainTest(TestCase):

    def test_assignment(self):
        assign_dom = AssignmentDomain(2)
        expected_dim = 4
        expected_dom = {(1, 0, 0, 1), (0, 1, 1, 0)}
        self.assertEqual(expected_dim, assign_dom.dim)
        self.assertEqual(expected_dom, set(assign_dom))

    def test_cube(self):
        cube_domain = CubeDomain(3)
        expected_dim = 3
        expected_dom = {(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0),
                        (1, 0, 1), (0, 1, 1), (1, 1, 1)}
        self.assertEqual(expected_dim, cube_domain.dim)
        self.assertEqual(expected_dom, set(cube_domain))


class ObjectivesTest(TestCase):

    def test_read(self):
        temp = NamedTemporaryFile(delete=False)
        temp.write(b'[3, 6, 4, 5, 2, 3, 5, 4, 3, 5, 4, 2, 4, 5, 3, 6]\n'
                   b'[2, 3, 5, 4, 5, 3, 4, 3, 5, 2, 6, 4, 4, 5, 2, 5]\n'
                   b'[4, 2, 4, 2, 4, 2, 4, 6, 4, 2, 6, 3, 2, 4, 5, 3]')
        temp.flush()
        temp.close()
        objs = Objectives.read(temp.name, delimiter=',')
        assert os.path.exists(temp.name)
        os.remove(temp.name)
        assert not os.path.exists(temp.name)
        objs2 = Objectives()
        objs2.obj = [3, 6, 4, 5, 2, 3, 5, 4, 3, 5, 4, 2, 4, 5, 3, 6]
        objs2.obj = [2, 3, 5, 4, 5, 3, 4, 3, 5, 2, 6, 4, 4, 5, 2, 5]
        objs2.obj = [4, 2, 4, 2, 4, 2, 4, 6, 4, 2, 6, 3, 2, 4, 5, 3]
        self.assertEqual(objs._objectives, objs2._objectives)

class PolyNondomTest(TestCase):

    def setUp(self):
        self.expected_nd_points = {(11, 11, 14), (15, 9, 17), (19, 14, 10),
                              (13, 16, 11), (15, 13, 13), (17, 15, 11),
                              (14, 14, 13)}
        self.expected_polynd_points = {(11, 11, 14), (15, 9, 17), (19,14,10),
                                  (13, 16, 11), (15, 13, 13)}
        self.expected_monond_points = {(17, 15, 11), (14, 14, 13)}
        self.expected_polynd_boxes = {((13, 15), (13, 16), (13, 14)),
                                 ((13, 19), (14, 16), (11, 14)),
                                 ((15, 19), (14, 16), (11, 17))}

    def test_read_points(self):
        temp = NamedTemporaryFile(delete=False)
        temp.write(b'[13, 16, 11], [18, 18, 14], [20, 17, 13],\n'
                   b'[18, 19, 15], [15, 9, 17], [16, 16, 15],\n'
                   b'[11, 11, 14], [16, 17, 12], [16, 20, 16],\n'
                   b'[13, 19, 15], [16, 13, 17], [15, 13, 13],\n'
                   b'[19, 13, 13], [16, 18, 13], [17, 15, 11],\n'
                   b'[15, 15, 15], [14, 14, 13], [16, 18, 18],\n'
                   b'[16, 16, 20], [18, 16, 16], [13, 14, 14],\n'
                   b'[17, 14, 14], [19, 14, 10], [17, 17, 13]')
        temp.flush()
        temp.close()
        ins = PolyNondom.read_points(temp.name, delimiter=',')
        self.assertEqual(ins.points['n'].points, self.expected_nd_points)
        self.assertEqual(ins.points['p'].points, self.expected_polynd_points)
        self.assertEqual(ins.points['m'].points, self.expected_monond_points)
        self.assertEqual(set(ins.polynd_boxes), self.expected_polynd_boxes)
        assert os.path.exists(temp.name)
        os.remove(temp.name)
        assert not os.path.exists(temp.name)

    def test_compute_points(self):
        assign_dom = AssignmentDomain(4)
        mult_objs = Objectives()
        mult_objs.obj = [3, 6, 4, 5, 2, 3, 5, 4, 3, 5, 4, 2, 4, 5, 3, 6]
        mult_objs.obj = [2, 3, 5, 4, 5, 3, 4, 3, 5, 2, 6, 4, 4, 5, 2, 5]
        mult_objs.obj = [4, 2, 4, 2, 4, 2, 4, 6, 4, 2, 6, 3, 2, 4, 5, 3]
        ins = PolyNondom.compute_points(assign_dom, mult_objs)
        self.assertEqual(ins.points['n'].points, self.expected_nd_points)
        self.assertEqual(ins.points['p'].points, self.expected_polynd_points)
        self.assertEqual(ins.points['m'].points, self.expected_monond_points)
        self.assertEqual(set(ins.polynd_boxes), self.expected_polynd_boxes)

    def test_is_dominated(self):
        a = (1, 1, 1, 1)
        b = (1, 1, 1, 2)
        c = (-1, 3, 4, 3)
        self.assertFalse(PolyNondom._is_dominated(a, b))
        self.assertTrue(PolyNondom._is_dominated(b, a))
        self.assertTrue(PolyNondom._is_dominated(b, a, 0))
        self.assertTrue(PolyNondom._is_dominated(b, a, 1))
        self.assertTrue(PolyNondom._is_dominated(b, a, 2))
        self.assertFalse(PolyNondom._is_dominated(b, a, 3))
        self.assertFalse(PolyNondom._is_dominated(a, c))
        self.assertFalse(PolyNondom._is_dominated(b, c))

    def test_is_nondominated(self):
        point1 = (1, 3, 4, 5)
        point2 = (5, 9, 0, 11)
        points = {(3, 1, 5, 4), (2, 2, 2, 3), (1, 9, 3, 0)}
        self.assertFalse(PolyNondom._is_nondominated(point1, points, 0))
        self.assertTrue(PolyNondom._is_nondominated(point2, points))

    def test_get_box(self):
        b1 = [19, 14, 10]
        b2 = [13, 16, 11]
        b3 = [15, 9, 17]
        expected_b = ((15, 19), (14, 16), (11, 17))
        self.assertEqual(expected_b, PolyNondom._get_box([b1, b2, b3]))
        c1 = [15, 13, 13]
        c2 = [13, 16, 11]
        c3 = [11, 11, 14]
        expected_c = ((13, 15), (13, 16), (13, 14))
        self.assertEqual(expected_c, PolyNondom._get_box((c1, c2, c3)))
        d1 = [15,13,13]
        d2 = [13,16,11]
        d3 = [15,9,17]
        self.assertRaises(InfeasibleBoxError, PolyNondom._get_box, [d1, d2, d3])


if __name__ == '__main__':
    main()
