(set-logic QF_NRA)
(declare-fun r_s () Real)
(declare-fun r_b () Real)
(declare-fun m () Real)
(declare-fun A () Real)
(declare-fun c_s () Real)
(declare-fun c_a () Real)
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
(declare-fun T_f () Real)
(declare-fun R_gas () Real)
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
(declare-fun K () Real)
(declare-fun L_sun () Real)
(declare-fun c () Real)
(declare-fun L () Real)
(declare-fun L_sha () Real)
(declare-fun DYL () Real)
(declare-fun DYL_max () Real)
(declare-fun f () Real)
(declare-fun f_T_v () Real)
(declare-fun f_DYL () Real)
(declare-fun decl () Real)
(declare-fun decl_max () Real)
(declare-fun lat () Real)
(declare-fun e_a () Real)
(declare-fun e_a_prime () Real)
(declare-fun e_i () Real)
(declare-fun q_s () Real)
(declare-fun active_bgc () Real)
(declare-fun beta_t () Real)
(declare-fun x () Real)

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

; 8.10
(= SLA (+ SLA_0 (* SLA_m x)))

; 8.11
(= SLA_sun (* (pow K -2) (pow L_sun -1) (+ SLA_m (* K SLA_0) (* -1 SLA_m c) (* -1 K SLA_0 c) (* -1 K L SLA_m c))))

; 8.12
(= SLA_sha (* (pow L_sha -1) (+ (* L (+ SLA_0 (* (/ 1 2) L SLA_m))) (* -1 L_sun SLA_sun))))

; 8.13

; 8.14
(= f_T_v (pow (+ 1 (exp (* 1000. (pow R_gas -1) (pow (+ T_f T_v) -1) (+ -220000 (* 710 T_f) (* 710 T_v))))) -1))
(= R_gas 8314.46759)
(= T_f 273.15)

; 8.15
(>= f_DYL 0.01) 
(<= f_DYL 1) 
(= f_DYL (* (pow DYL 2) (pow DYL_max -2)))

; 8.16
(= DYL (* 27501.9742 (arccos (* (pow (cos decl) -1) (pow (cos lat) -1) (sin decl) (sin lat)))))
(= DYL_max (* 27501.9742 (arccos (* (pow (cos decl_max) -1) (pow (cos lat) -1) (sin decl_max) (sin lat)))))
(= decl_max 23.4667)

; 8.17

; 8.18

; 8.19

; 8.20

; 8.21

; 8.22
(= A (* (pow P_atm -1) (pow (+ (* 1.37 r_b) (* 1.65 r_s)) -1) (+ c_a (* -1 c_i))))
(= A (* 0.72992700729927 (pow P_atm -1) (pow r_b -1) (+ c_a (* -1 c_s))))
(= A (* 0.606060606060606 (pow P_atm -1) (pow r_s -1) (+ c_s (* -1 c_i))))

; 8.23
(= (* (pow (+ r_b r_s) -1) (+ e_a_prime (* -1 e_i))) (* (pow r_b -1) (+ e_a_prime (* -1 e_s))))
(= (* (pow r_b -1) (+ e_a_prime (* -1 e_s))) (* (pow r_s -1) (+ e_s (* -1 e_i))))

; 8.24
(= e_a (* 1.60771704180064 P_atm q_s))

; 8.25
(= c_s (+ c_a (* -1.37 A P_atm r_b)))

; 8.26
(= e_s (* (pow (+ r_b r_s) -1) (+ (* e_a_prime r_s) (* e_i r_b))))

; 8.27
(= (+ (* -1 r_b) (* r_s (+ -1 (* b r_b) (* A P_atm m r_b (pow c_s -1)))) (* (pow r_s 2) (+ b (* A P_atm e_a_prime m (pow c_s -1) (pow e_i -1))))) 0)

; 8.28
(= c_i (+ c_s (* -1.65 A P_atm r_s)))

))
(push 1)
(check-sat)
