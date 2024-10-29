(define (domain pick-and-place)
  (:requirements :strips :equality :action-costs)
  (:types wuti grasp_dir grasp config pose trajectory)
  (:predicates

    (graspable ?o - wuti)
    (attached ?o - wuti)
    (handempty)
    (on ?o - wuti ?r - wuti)

    (occupied ?r - wuti)

  )

  (:functions
      (total-cost) - number
  )


  (:action pick
    :parameters (?o - wuti)
    :precondition (and (handempty)
                       (graspable ?o)
                       (not (attached ?o))
                       (not (occupied ?o))
                       )
    :effect (and (not (handempty))
                 (attached ?o)
                 (forall (?r - wuti) (not (on ?o ?r)))
                 (increase (total-cost) 100)
                 )
  )


  (:action place
    :parameters (?o - wuti ?r - wuti)
    :precondition (and (attached ?o)
                       (not (= ?o ?r))
                       )
    :effect (and (handempty)
                 (not (attached ?o))
                 (on ?o ?r)
                 (increase (total-cost) 100)
                 )
  )


  (:derived (occupied ?r - wuti)
    (exists (?o) (on ?o ?r))
  )


)