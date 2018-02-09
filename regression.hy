(import [numpy :as np]
        [tensorflow :as tf])

(setv num_points 1000)
(setv vectors_set [])

(for [i (range num_points)] 
  (setv x1 (np.random.normal 0.0 0.03))
  (setv y1 (+ (* x1 0.1) 0.3 (np.random.normal 0.0 0.03)))
  (.append vectors_set [x1 y1]))

(setv x_data (list-comp (get v 0) [v vectors_set]))
(setv y_data (list-comp (get v 1) [v vectors_set]))

(setv W (tf.Variable (tf.random_uniform [1] -1.0 1.0)))
(setv b (tf.Variable (tf.zeros [1])))
(setv y (+ (* W x_data) b))

(setv loss (tf.reduce_mean (tf.square (- y y_data))))
(setv optimizer (tf.train.GradientDescentOptimizer 0.5))
(setv train (optimizer.minimize loss))

(setv init (tf.global_variables_initializer))

(setv sess (tf.Session))
(sess.run init)

(for [step (range 101)]
  (sess.run train)
  (if (or (= (% step 20) 0) (< step 8))
    (do (print step (sess.run W) (sess.run b))
        (print step (sess.run loss)))))
