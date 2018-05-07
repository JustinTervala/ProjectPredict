from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)

import daft


def make_sample_project():
    pgm = daft.PGM([3, 3])

    pgm.add_node(daft.Node("A", "A", 0.5, 2))
    pgm.add_node(daft.Node("B", "B", 0.5, 1))
    pgm.add_node(daft.Node("C", "C", 1.5, 1.5))
    pgm.add_node(daft.Node("D", "D", 1.5, 0.5))
    pgm.add_node(daft.Node("E", "E", 2.5, 2))
    pgm.add_node(daft.Node("F", "F", 2.5, 1))

    pgm.add_edge("A", "C")
    pgm.add_edge("B", "C")
    pgm.add_edge("B", "D")
    pgm.add_edge("C", "E")
    pgm.add_edge("C", "F")

    pgm.render()
    pgm.figure.savefig("sample_project.png", dpi=300)


def make_task_bn():
    pgm = daft.PGM([5, 3])

    pgm.add_node(daft.Node("duration", r"$D$", 2.5, 2.5))
    pgm.add_node(daft.Node("earliest start", r"$ES$", 4.5, 1.5, observed=True))
    pgm.add_node(daft.Node("latest start", r"$LS$", 0.5, 1.5))
    pgm.add_node(daft.Node("earliest finish", r"$EF$", 3.5, 0.5))
    pgm.add_node(daft.Node("latest finish", r"$LF$", 1.5, 0.5, observed=True))

    pgm.add_edge("duration", "latest start")
    pgm.add_edge("duration", "earliest finish")
    pgm.add_edge("earliest start", "earliest finish")
    pgm.add_edge("latest finish", "latest start")

    pgm.render()
    pgm.figure.savefig("task_bn.png", dpi=300)


if __name__ == '__main__':
    make_sample_project()
    make_task_bn()
