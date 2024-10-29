; test 0921

(define (problem put-wuti)
   (:domain pick-and-place)

   (:objects
          o1 o3 r01 r02 r11 r12 - wuti
   )

   (:init
          (graspable o1)
          (graspable o3)
          (handempty)
          (on o1 r12)
          (on o3 r01)
          (= (total-cost) 0)
   )

   (:goal
        (and (on o3 r02))
   )

   (:metric minimize (total-cost))

)
