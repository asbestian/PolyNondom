Glossary
============

.. _nd-label:

non-dominated point 
    A point *y* is non-dominated (in the point set P) if there is no other 
    point *z* in P such that *z_i <= y_i* for all i with at least one 
    strict inequality.


.. _dom-label:

dominated point
    A point is *y* is dominated if *y* is not non-dominated.

.. _polynd-label:

polynon-dominated point
    A non-dominated point *y* is polynon-dominated (w.r.t. the underlying
    multi-objective problem) if the projection of the point is also
    non-dominated for the given multi-objective problem where one of
    the objectives was neglected

.. _monond-label:

mononon-dominated point
    A non-dominated point *y* is mononon-dominated if *y* is not 
    polynon-dominated.
  
For more details about the definitions see this ZIB-Report_.

.. _ZIB-Report: https://opus4.kobv.de/opus4-zib/files/6128/report_16-55.pdf
