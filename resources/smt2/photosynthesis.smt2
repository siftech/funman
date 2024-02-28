(set-logic QF_NRA)
(declare-fun r_s () Real)
(declare-fun m () Real)
(declare-fun A () Real)
(declare-fun c_s () Real)
(declare-fun e_s () Real)
(declare-fun e_t () Real)
(declare-fun P_atm () Real)
(declare-fun b () Real)
(declare-fun C_3 () Bool)
(declare-fun C_4 () Bool)

(declare-fun w_c () Real)
(declare-fun w_j () Real)
(declare-fun w_e () Real)
(declare-fun Gamma_star () Real)
(declare-fun V_cmax () Real)
(declare-fun Gamma_star () Real)
(declare-fun K_c () Real)
(declare-fun o_i () Real)
(declare-fun K_o () Real)
(declare-fun phi () Real)
(declare-fun alpha () Real)
(declare-fun c_i () Real)
(declare-fun K_c25 () Real)
(declare-fun a_kc () Real)
(declare-fun T_v () Real)
(declare-fun K_o25 () Real)
(declare-fun a_ko () Real)
(declare-fun V_cmax25 () Real)
(declare-fun N_a () Real)
(declare-fun F_LNR () Real)
(declare-fun F_NR () Real)
(declare-fun a_R25 () Real)
(declare-fun CN_L () Real)
(declare-fun SLA_0 () Real)
(declare-fun SLA_m () Real)
(declare-fun SLA () Real)
(declare-fun SLA_sun () Real)
(declare-fun SLA_sha () Real)





(assert (and 

; 8.1
(= (/ 1 r_s) (+ (* m (/ A c_s) (/ e_s e_t) P_atm) b))
(=> (= A 0) (= b 2000))
(=> C_3 (= m 9))
(=> C_4 (= m 4))
(xor C_3 C_4)

; 8.2
;(= A (minimum w_c w_j w_e))
(= C_3 (= w_c (/ (* V_cmax (- c_i Gamma_star)) (+ c_i (* K_c (/ (+ 1 o_i) K_o))))))
(= C_4 (= w_c V_cmax))
(<= 0 (- c_i Gamma_star))

; 8.3
(= C_3 (= w_j (/ (* (- c_i Gamma_star) 4.6 phi alpha) (+ c_i (* 2 Gamma_star)))))
(= C_4 (= w_j (* 4.6 phi alpha)))

; 8.4
(= C_3 (= w_e (* 0.5 V_cmax)))
(= C_4 (= w_e (* 4000 V_cmax (/ c_i P_atm))))
(= o_i (+ 0.209 P_atm))

; 8.5 
(= K_c (* K_c25 (pow a_kc (/ (- T_v 25) 10))))
(= K_c25 30.0)
(= a_kc 2.1)

; 8.6
(= K_o (* K_o25 (pow a_ko (/ (- T_v 25) 10))))
(= K_o25 30000.0)
(= a_ko 1.2)

; 8.7
(= Gamma_star (* 0.5 (/ K_c K_o) 0.21 o_i))

; 8.8
(= V_cmax25 (* N_a F_LNR F_NR a_R25))
(= F_NR 7.16)
(= a_R25 60)

; 8.9
(= N_a (/ 1 (* CN_L SLA)))


(= SLA )

))
(push 1)
(check-sat)