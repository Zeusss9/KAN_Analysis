import numpy as np
import torch
from sklearn.linear_model import LinearRegression
import sympy

# sigmoid = sympy.Function('sigmoid')
# name: (torch implementation, sympy implementation)

def sigmoid(x):
    return 1 / (1 + sympy.exp(-x))

def relu(x):
    return sympy.Max(0, x)

SYMBOLIC_LIB = {'x': (lambda x: x, lambda x: x),
                 'x^2': (lambda x: x**2, lambda x: x**2),
                 'x^3': (lambda x: x**3, lambda x: x**3),
                 'x^4': (lambda x: x**4, lambda x: x**4),
                 '1/x': (lambda x: 1/x, lambda x: 1/x),
                 '1/x^2': (lambda x: 1/x**2, lambda x: 1/x**2),
                 '1/x^3': (lambda x: 1/x**3, lambda x: 1/x**3),
                 '1/x^4': (lambda x: 1/x**4, lambda x: 1/x**4),
                 'sqrt': (lambda x: torch.sqrt(x), lambda x: sympy.sqrt(x)),
                 '1/sqrt(x)': (lambda x: 1/torch.sqrt(x), lambda x: 1/sympy.sqrt(x)),
                 'exp': (lambda x: torch.exp(x), lambda x: sympy.exp(x)),
                 'log': (lambda x: torch.log(x), lambda x: sympy.log(x)),
                 'abs': (lambda x: torch.abs(x), lambda x: sympy.Abs(x)),
                 'sin': (lambda x: torch.sin(x), lambda x: sympy.sin(x)),
                 'tan': (lambda x: torch.tan(x), lambda x: sympy.tan(x)),
                 'tanh': (lambda x: torch.tanh(x), lambda x: sympy.tanh(x)),
                 'sigmoid': (lambda x: torch.sigmoid(x), lambda x: sigmoid(x)),
                 'relu': (lambda x: torch.relu(x), lambda x: relu(x)),
                 'sgn': (lambda x: torch.sign(x), lambda x: sympy.sign(x)),
                 'arcsin': (lambda x: torch.arcsin(x), lambda x: sympy.asin(x)),
                 'arctan': (lambda x: torch.arctan(x), lambda x: sympy.atan(x)),
                 'arctanh': (lambda x: torch.arctanh(x), lambda x: sympy.atanh(x)),
                 '0': (lambda x: x*0, lambda x: x*0),
                 'gaussian': (lambda x: torch.exp(-x**2), lambda x: sympy.exp(-x**2)),
                 'cosh': (lambda x: torch.cosh(x), lambda x: sympy.cosh(x)),

                 'custom_0': (lambda x: x - 1.07142275513362*torch.sin(x + torch.sin(x - 0.283036) + 0.19679742) - 0.11781596, lambda x: x - 1.07142275513362*sympy.sin(x + sympy.sin(x - 0.283036) + 0.19679742) - 0.11781596),
                 'custom_1': (lambda x: (-x + torch.sqrt(torch.exp(torch.arctan(2.87648599266381*x))) - 1.4587086)*torch.abs(torch.cos(x + torch.cos(0.41750595*x))), lambda x: (-x + sympy.sqrt(sympy.exp(sympy.atan(2.87648599266381*x))) - 1.4587086)*sympy.Abs(sympy.cos(x + sympy.cos(0.41750595*x)))),
                 'custom_2': (lambda x: torch.log(torch.log(0.2359558*torch.sin(0.906780422715593*torch.exp(-torch.asinh(torch.asinh(x + 1.2268976)/torch.cos(torch.cos(torch.exp(x)))))) + 1) + 1), lambda x: sympy.log(sympy.log(0.2359558*sympy.sin(0.906780422715593*sympy.exp(-sympy.asinh(sympy.asinh(x + 1.2268976)/sympy.cos(sympy.cos(sympy.exp(x)))))) + 1) + 1)),
                 'custom_3': (lambda x: -torch.exp(0.40566453*x)*torch.sin(0.26832688*torch.cos(1.31389884240256*x + 0.327405302054833) + 0.0157287287031952), lambda x: -sympy.exp(0.40566453*x)*sympy.sin(0.26832688*sympy.cos(1.31389884240256*x + 0.327405302054833) + 0.0157287287031952)),
                 'custom_4': (lambda x: -torch.tanh(0.6736647*(x**2/torch.sqrt(x**4 + 1) - 0.60106015)*torch.tanh(torch.sinh(x) + 0.68494785) + 0.117319934), lambda x: -sympy.tanh(0.6736647*(x**2/sympy.sqrt(x**4 + 1) - 0.60106015)*sympy.tanh(sympy.sinh(x) + 0.68494785) + 0.117319934)),
                 'custom_5': (lambda x: -0.07707365*torch.sinh(torch.tan(torch.tanh(torch.tanh(x)*torch.abs(torch.tan(torch.cos(x + 0.25272873)))) + 0.13659722)), lambda x: -0.07707365*sympy.sinh(sympy.tan(sympy.tanh(sympy.tanh(x)*sympy.Abs(sympy.tan(sympy.cos(x + 0.25272873)))) + 0.13659722))),
                 'custom_6': (lambda x: (0.5515897*torch.sin(0.493899989400906*x) + 0.23895919340327)*torch.log(torch.sin(x) + 1.42061475), lambda x: (0.5515897*sympy.sin(0.493899989400906*x) + 0.23895919340327)*sympy.log(sympy.sin(x) + 1.42061475)),
                 'custom_7': (lambda x: -torch.tan(0.307751131327064*torch.cos(x - 0.29171234)**2*torch.atan(x)**3 + 0.08414273), lambda x: -sympy.tan(0.307751131327064*sympy.cos(x - 0.29171234)**2*sympy.atan(x)**3 + 0.08414273)),
                 'custom_8': (lambda x: torch.tan(1.7573195787227*torch.atan(torch.sin((-0.089042924*x - torch.sin(0.2408634*x)**3 - 0.5916287)**3)) + 3.02046788774369), lambda x: sympy.tan(1.7573195787227*sympy.atan(sympy.sin((-0.089042924*x - sympy.sin(0.2408634*x)**3 - 0.5916287)**3)) + 3.02046788774369)),
                 'custom_9': (lambda x: -x - 1.2073365*torch.atan(x) + (-torch.cos(x) - 0.96062267)/(0.25869083 - 0.025811067*x), lambda x: -x - 1.2073365*sympy.atan(x) + (-sympy.cos(x) - 0.96062267)/(0.25869083 - 0.025811067*x)),
                 'custom_10': (lambda x: torch.sin(torch.abs(torch.log(torch.asinh(torch.cos(torch.log(torch.abs(x - 0.4102029) + 1))) + 1)**3 + 0.14048469) - 0.109476835), lambda x: sympy.sin(sympy.Abs(sympy.log(sympy.asinh(sympy.cos(sympy.log(sympy.Abs(x - 0.4102029) + 1))) + 1)**3 + 0.14048469) - 0.109476835)),
                 'custom_11': (lambda x: -x*torch.sin(0.91479486*x) + torch.tan(0.0066639213*torch.exp(x)) + 4.9237704*torch.atan(x) + 1.7823888, lambda x: -x*sympy.sin(0.91479486*x) + sympy.tan(0.0066639213*sympy.exp(x)) + 4.9237704*sympy.atan(x) + 1.7823888),
                 'custom_12': (lambda x: 1.54825853737626*x + 1.82459000736087*torch.sin(torch.sin(x)) + 3.97855878257996*torch.cos(1.0611413*x) - 0.0393361138611623, lambda x: 1.54825853737626*x + 1.82459000736087*sympy.sin(sympy.sin(x)) + 3.97855878257996*sympy.cos(1.0611413*x) - 0.0393361138611623),
                 'custom_13': (lambda x: torch.tan(torch.tanh(torch.asinh(x) + 0.639887)**18 - 0.041265465)*torch.tanh(torch.cosh(torch.cosh(torch.sin(x)))), lambda x: sympy.tan(sympy.tanh(sympy.asinh(x) + 0.639887)**18 - 0.041265465)*sympy.tanh(sympy.cosh(sympy.cosh(sympy.sin(x))))),
                 'custom_14': (lambda x: torch.tan(torch.asinh(torch.atan(0.44066956*x + 0.22033478*torch.sin(0.9295771*x)**4 + 0.120729776650162)**2 - 0.062562145)), lambda x: sympy.tan(sympy.asinh(sympy.atan(0.44066956*x + 0.22033478*sympy.sin(0.9295771*x)**4 + 0.120729776650162)**2 - 0.062562145))),
                 'custom_15': (lambda x: 0.18928082 - torch.atan(0.62086*torch.abs(x + torch.sinh(torch.sin(torch.sin(torch.atan(0.7841746*torch.sinh(x - 2.1201138))**2)))))**3, lambda x: 0.18928082 - sympy.atan(0.62086*sympy.Abs(x + sympy.sinh(sympy.sin(sympy.sin(sympy.atan(0.7841746*sympy.sinh(x - 2.1201138))**2)))))**3),
                 'custom_16': (lambda x: -torch.sin(torch.sin(x - 1.2395611))**20 - torch.tan(torch.sinh(torch.tanh(torch.asinh(x) + 0.0895923)**7)), lambda x: -sympy.sin(sympy.sin(x - 1.2395611))**20 - sympy.tan(sympy.sinh(sympy.tanh(sympy.asinh(x) + 0.0895923)**7))),
                 'custom_17': (lambda x: torch.tan((torch.asinh(torch.tanh(torch.sin(0.938381286650241*torch.tan(torch.sin(x - 0.13431023))))) + 0.17231958)*torch.cos(torch.asinh(x))), lambda x: sympy.tan((sympy.asinh(sympy.tanh(sympy.sin(0.938381286650241*sympy.tan(sympy.sin(x - 0.13431023))))) + 0.17231958)*sympy.cos(sympy.asinh(x)))),
 
                 'custom_19': (lambda x: torch.asinh(0.31529173*torch.exp(x) - 0.31529173*torch.sinh(torch.cos(0.9708182*torch.sinh(torch.cos(1.12777409872805*x - 0.193743582965378))))), lambda x: sympy.asinh(0.31529173*sympy.exp(x) - 0.31529173*sympy.sinh(sympy.cos(0.9708182*sympy.sinh(sympy.cos(1.12777409872805*x - 0.193743582965378)))))),
                 'custom_20': (lambda x: -0.0706494*torch.sinh(x) + 0.0706494*torch.cosh(x) + torch.tanh(torch.sinh(torch.sinh(1.30043023790095*x))) + 0.0706494*torch.asinh(x**2)**3, lambda x: -0.0706494*sympy.sinh(x) + 0.0706494*sympy.cosh(x) + sympy.tanh(sympy.sinh(sympy.sinh(1.30043023790095*x))) + 0.0706494*sympy.asinh(x**2)**3),
                #  'custom_21': (lambda x: torch.tan(x*torch.tan(torch.tan(torch.log(torch.cos(torch.tanh(1.88868388037808*torch.sqrt(0.0186486189689872*torch.exp(3*x) + 1)/x)**2))/torch.log(10)))), lambda x: sympy.tan(x*sympy.tan(sympy.tan(sympy.log(sympy.cos(sympy.tanh(1.88868388037808*sympy.sqrt(0.0186486189689872*sympy.exp(3*x) + 1)/x)**2))/sympy.log(10))))),
                 'custom_22': (lambda x: 2.6468036405196*(torch.abs(x + 4.936872) - 1.1953378)*torch.atan(0.578082566029891*x) - 1.644881, lambda x: 2.6468036405196*(sympy.Abs(x + 4.936872) - 1.1953378)*sympy.atan(0.578082566029891*x) - 1.644881),
                 'custom_23': (lambda x: -1.06766258232346*x*torch.sin(torch.atan(1.02839506027596*torch.sqrt(torch.exp(0.9366255*x)))**2 - 0.09030407) + 0.263440030193498, lambda x: -1.06766258232346*x*sympy.sin(sympy.atan(1.02839506027596*sympy.sqrt(sympy.exp(0.9366255*x)))**2 - 0.09030407) + 0.263440030193498),
                #  'custom_24': (lambda x: -x - 5.64489862044323*torch.cos(0.89212614*x - 0.89212614*torch.log(torch.atan(torch.exp(torch.cosh(x))))/torch.log(10)) - 1.6864482, lambda x: -x - 5.64489862044323*sympy.cos(0.89212614*x - 0.89212614*sympy.log(sympy.atan(sympy.exp(sympy.cosh(x))))/sympy.log(10)) - 1.6864482),
                 'custom_25': (lambda x: (x - 5.5881186)*(torch.sin(0.9661117*x + 0.409545608725508) - 1.233576) - 8.506843, lambda x: (x - 5.5881186)*(sympy.sin(0.9661117*x + 0.409545608725508) - 1.233576) - 8.506843),

                # 1,0,0
                'custom_26': (lambda x: (0.9730727 - torch.sin(torch.sin(0.72740996*x - 0.4604782)))*(3.3033967 - 2*x) + torch.sin(x) - 3.8944724, lambda x: (0.9730727 - sympy.sin(sympy.sin(0.72740996*x - 0.4604782)))*(3.3033967 - 2*x) + sympy.sin(x) - 3.8944724),
                # 1,0,2
                'custom_27': (lambda x: -0.09398928*torch.exp(x) - 0.09398928*torch.sin(2*x) - 8.54963920522554*torch.sin(0.762242181110267*x - 1.0152711) + 1.0148481, lambda x: -0.09398928*sympy.exp(x) - 0.09398928*sympy.sin(2*x) - 8.54963920522554*sympy.sin(0.762242181110267*x - 1.0152711) + 1.0148481),
                # 0,8,1
                'custom_28': (lambda x: 0.387830153507441*torch.sin(0.7821311*x - torch.tan(torch.sin(0.8138459*x) - 0.04185148)) + 0.0555350789852952, lambda x: 0.387830153507441*sympy.sin(0.7821311*x - sympy.tan(sympy.sin(0.8138459*x) - 0.04185148)) + 0.0555350789852952),
                # 0,14,1
                'custom_29': (lambda x: (x + 3.0300636)*(-0.009821785*x - 0.009821785*torch.sin(2.85705485985317*x + 0.4405821) + 0.00123982268015935), lambda x: (x + 3.0300636)*(-0.009821785*x - 0.009821785*sympy.sin(2.85705485985317*x + 0.4405821) + 0.00123982268015935)),
                # 0,8,0
                'custom_30': (lambda x: -torch.tan(0.08911767*x + 0.23142147*torch.sin(-1.332530723428*x + 1.332530723428*torch.sin(x - 0.33004436) + 0.915394506247664) - 0.0246020619243789), lambda x: -sympy.tan(0.08911767*x + 0.23142147*sympy.sin(-1.332530723428*x + 1.332530723428*sympy.sin(x - 0.33004436) + 0.915394506247664) - 0.0246020619243789)),
                # # 0,9,0
                # 'custom_31': (lambda x: , lambda x: ),
                # # 0,12,0
                # 'custom_32': (lambda x: , lambda x: ),
                # # 0,15,1
                # 'custom_33': (lambda x: , lambda x: ),
                # 1,1,5
                'custom_33': (lambda x: 0.733569151398909*x*(x + torch.sin(0.91680056*x + 0.19586428) + 3.4290361) - 0.34579208, lambda x: 0.733569151398909*x*(x + sympy.sin(0.91680056*x + 0.19586428) + 3.4290361) - 0.34579208),             
                                                
                #  'custom_0': (lambda x: -2.09492501e-04*(x**10)+3.78684997e-03*(x**9)-2.38853669e-02*(x**8)+4.57417382e-02*(x**7)+1.15283213e-01*(x**6)-4.96508120e-01*(x**5)-3.87796197e-02*(x**4)+1.56554178e+00*(x**3)-1.32581377e-01*(x**2)-1.13996357e+00*x-4.44868232e-02, lambda x: -2.09492501e-04*(x**10)+3.78684997e-03*(x**9)-2.38853669e-02*(x**8)+4.57417382e-02*(x**7)+1.15283213e-01*(x**6)-4.96508120e-01*(x**5)-3.87796197e-02*(x**4)+1.56554178e+00*(x**3)-1.32581377e-01*(x**2)-1.13996357e+00*x-4.44868232e-02),
                #  'custom_1': (lambda x: -0.00110631*(x**10)+0.00613431*(x**9)+0.00755025*(x**8)-0.08368167*(x**7)+0.01781185*(x**6)+0.43334373*(x**5)-0.25285241*(x**4)-1.02114736*(x**3)+0.46259957*(x**2)+0.53395778*x-0.24924803, lambda x: -0.00110631*(x**10)+0.00613431*(x**9)+0.00755025*(x**8)-0.08368167*(x**7)+0.01781185*(x**6)+0.43334373*(x**5)-0.25285241*(x**4)-1.02114736*(x**3)+0.46259957*(x**2)+0.53395778*x-0.24924803),
                #  'custom_2': (lambda x: 0.00012995*(x**10)+0.00113579*(x**9)+0.00102224*(x**8)-0.01238183*(x**7)-0.01708189*(x**6)+0.05682448*(x**5)+0.05235142*(x**4)-0.10503659*(x**3)-0.0301626*(x**2)+0.0124743*x+0.07600713, lambda x: 0.00012995*(x**10)+0.00113579*(x**9)+0.00102224*(x**8)-0.01238183*(x**7)-0.01708189*(x**6)+0.05682448*(x**5)+0.05235142*(x**4)-0.10503659*(x**3)-0.0301626*(x**2)+0.0124743*x+0.07600713),
                #  'custom_3': (lambda x: -0.00063381*(x**6)-0.00754891*(x**5)-0.0190362*(x**4)+0.05864355*(x**3)+0.23092394*(x**2)+0.0053303*x-0.26534153, lambda x: -0.00063381*(x**6)-0.00754891*(x**5)-0.0190362*(x**4)+0.05864355*(x**3)+0.23092394*(x**2)+0.0053303*x-0.26534153),
                #  'custom_4': (lambda x: 1.29876016e-04*(x**12)-3.06614604e-04*(x**11)-3.02782271e-03*(x**10)+6.55328122e-03*(x**9)+2.93860674e-02*(x**8)-5.61720260e-02*(x**7)-1.51340938e-01*(x**6)+2.41675660e-01*(x**5)+4.21709055e-01*(x**4)-5.06290904e-01*(x**3)-5.57891942e-01*(x**2)+2.72219285e-01*x+1.20627901e-01, lambda x: 1.29876016e-04*(x**12)-3.06614604e-04*(x**11)-3.02782271e-03*(x**10)+6.55328122e-03*(x**9)+2.93860674e-02*(x**8)-5.61720260e-02*(x**7)-1.51340938e-01*(x**6)+2.41675660e-01*(x**5)+4.21709055e-01*(x**4)-5.06290904e-01*(x**3)-5.57891942e-01*(x**2)+2.72219285e-01*x+1.20627901e-01),
                #  'custom_5': (lambda x: 4.35646459e-05*(x**11)-2.57378275e-05*(x**10)-1.22730713e-03*(x**9)+5.65125640e-04*(x**8)+1.27182960e-02*(x**7)-2.53600865e-03*(x**6)-6.17934553e-02*(x**5)-4.67611699e-03*(x**4)+1.34664247e-01*(x**3)+2.11019737e-02*(x**2)-1.16256660e-01*x-9.53366610e-03, lambda x: 4.35646459e-05*(x**11)-2.57378275e-05*(x**10)-1.22730713e-03*(x**9)+5.65125640e-04*(x**8)+1.27182960e-02*(x**7)-2.53600865e-03*(x**6)-6.17934553e-02*(x**5)-4.67611699e-03*(x**4)+1.34664247e-01*(x**3)+2.11019737e-02*(x**2)-1.16256660e-01*x-9.53366610e-03),
                #  'custom_6': (lambda x: -1.38987852e-04*(x**9)-1.80108957e-03*(x**8)-6.32635210e-03*(x**7)+4.78527536e-03*(x**6)+4.77838763e-02*(x**5)-1.07699891e-02*(x**4)-1.38250812e-01*(x**3)+1.30609983e-01*(x**2)+2.91683084e-01*x+8.16843333e-02, lambda x: -1.38987852e-04*(x**9)-1.80108957e-03*(x**8)-6.32635210e-03*(x**7)+4.78527536e-03*(x**6)+4.77838763e-02*(x**5)-1.07699891e-02*(x**4)-1.38250812e-01*(x**3)+1.30609983e-01*(x**2)+2.91683084e-01*x+8.16843333e-02),
                #  'custom_7': (lambda x: 5.39425085e-05*(x**11)+3.62791589e-04*(x**10)-2.53101575e-04*(x**9)-4.69610823e-03*(x**8)-6.49230856e-05*(x**7)+2.06828187e-02*(x**6)-8.47869013e-03*(x**5)-1.76599048e-02*(x**4)+4.45352811e-02*(x**3)-3.11489812e-02*(x**2)-7.61330554e-02*x-8.46062797e-02, lambda x: 5.39425085e-05*(x**11)+3.62791589e-04*(x**10)-2.53101575e-04*(x**9)-4.69610823e-03*(x**8)-6.49230856e-05*(x**7)+2.06828187e-02*(x**6)-8.47869013e-03*(x**5)-1.76599048e-02*(x**4)+4.45352811e-02*(x**3)-3.11489812e-02*(x**2)-7.61330554e-02*x-8.46062797e-02),
                #  'custom_8': (lambda x: 3.17095289e-05*(x**8)+7.65169946e-04*(x**7)+6.45992383e-03*(x**6)+2.12888041e-02*(x**5)+1.38478261e-02*(x**4)-5.01687825e-02*(x**3)-8.21463054e-02*(x**2)-1.96932742e-01*x-5.09007731e-01, lambda x: 3.17095289e-05*(x**8)+7.65169946e-04*(x**7)+6.45992383e-03*(x**6)+2.12888041e-02*(x**5)+1.38478261e-02*(x**4)-5.01687825e-02*(x**3)-8.21463054e-02*(x**2)-1.96932742e-01*x-5.09007731e-01),
                #  'custom_9': (lambda x: -7.58852736e-06*(x**9)-4.65089364e-05*(x**8)+6.47458992e-04*(x**7)+4.10806977e-03*(x**6)-2.18871980e-02*(x**5)-1.44152011e-01*(x**4)+3.04164854e-01*(x**3)+1.85945674e+00*(x**2)-2.82622768e+00*x-7.58836898e+00, lambda x: -7.58852736e-06*(x**9)-4.65089364e-05*(x**8)+6.47458992e-04*(x**7)+4.10806977e-03*(x**6)-2.18871980e-02*(x**5)-1.44152011e-01*(x**4)+3.04164854e-01*(x**3)+1.85945674e+00*(x**2)-2.82622768e+00*x-7.58836898e+00),
                #  'custom_10': (lambda x: 3.40548896e-05*(x**7)+6.79870615e-04*(x**6)+4.00532074e-03*(x**5)+2.30901099e-03*(x**4)-4.05152358e-02*(x**3)-7.40570902e-02*(x**2)+8.86678209e-02*x+2.55317031e-01, lambda x: 3.40548896e-05*(x**7)+6.79870615e-04*(x**6)+4.00532074e-03*(x**5)+2.30901099e-03*(x**4)-4.05152358e-02*(x**3)-7.40570902e-02*(x**2)+8.86678209e-02*x+2.55317031e-01),
                #  'custom_11': (lambda x: 1.25589571e-05*(x**9)+5.85574165e-05*(x**8)-1.09944035e-03*(x**7)-4.66735240e-03*(x**6)+3.76741936e-02*(x**5)+1.39416307e-01*(x**4)-5.28518706e-01*(x**3)-1.05688661e+00*(x**2)+4.42340774e+00*x+1.91964823e+00, lambda x: 1.25589571e-05*(x**9)+5.85574165e-05*(x**8)-1.09944035e-03*(x**7)-4.66735240e-03*(x**6)+3.76741936e-02*(x**5)+1.39416307e-01*(x**4)-5.28518706e-01*(x**3)-1.05688661e+00*(x**2)+4.42340774e+00*x+1.91964823e+00),
                #  'custom_12': (lambda x: 3.39698685e-04*(x**8)-1.09215211e-03*(x**7)-1.35594757e-02*(x**6)+2.80759848e-02*(x**5)+2.53482626e-01*(x**4)-3.38289035e-01*(x**3)-2.32384744e+00*(x**2)+3.20469397e+00*x+3.96972995e+00, lambda x: 3.39698685e-04*(x**8)-1.09215211e-03*(x**7)-1.35594757e-02*(x**6)+2.80759848e-02*(x**5)+2.53482626e-01*(x**4)-3.38289035e-01*(x**3)-2.32384744e+00*(x**2)+3.20469397e+00*x+3.96972995e+00),
                #  'custom_13': (lambda x: -0.00073535*(x**7)+0.0074761*(x**6)-0.01619832*(x**5)-0.03971497*(x**4)+0.1101441*(x**3)+0.13992093*(x**2)-0.01147367*x-0.05874407, lambda x: -0.00073535*(x**7)+0.0074761*(x**6)-0.01619832*(x**5)-0.03971497*(x**4)+0.1101441*(x**3)+0.13992093*(x**2)-0.01147367*x-0.05874407),
                #  'custom_14': (lambda x: 1.81060935e-04*(x**8)-2.91399651e-03*(x**7)+1.48804058e-02*(x**6)-1.48846080e-02*(x**5)-7.12639955e-02*(x**4)+8.92627643e-02*(x**3)+2.41978888e-01*(x**2)+7.73578944e-02*x-5.62590569e-02, lambda x: 1.81060935e-04*(x**8)-2.91399651e-03*(x**7)+1.48804058e-02*(x**6)-1.48846080e-02*(x**5)-7.12639955e-02*(x**4)+8.92627643e-02*(x**3)+2.41978888e-01*(x**2)+7.73578944e-02*x-5.62590569e-02),
                #  'custom_15': (lambda x: -8.05817056e-05*(x**9)+9.95087205e-04*(x**8)-2.23347099e-03*(x**7)-1.53507824e-02*(x**6)+6.72056508e-02*(x**5)+5.65024978e-03*(x**4)-2.61797924e-01*(x**3)+3.07797845e-03*(x**2)+1.05442460e-01*x-1.25633887e-02, lambda x: -8.05817056e-05*(x**9)+9.95087205e-04*(x**8)-2.23347099e-03*(x**7)-1.53507824e-02*(x**6)+6.72056508e-02*(x**5)+5.65024978e-03*(x**4)-2.61797924e-01*(x**3)+3.07797845e-03*(x**2)+1.05442460e-01*x-1.25633887e-02),
                #  'custom_16': (lambda x: 9.01107333e-05*(x**10)-1.72559454e-03*(x**9)+1.19035298e-02*(x**8)-2.94584036e-02*(x**7)-2.79889630e-02*(x**6)+2.19600898e-01*(x**5)-4.96161512e-02*(x**4)-6.68335614e-01*(x**3)+1.31055577e-01*(x**2)+9.21971231e-01*x+4.55105144e-02, lambda x: 9.01107333e-05*(x**10)-1.72559454e-03*(x**9)+1.19035298e-02*(x**8)-2.94584036e-02*(x**7)-2.79889630e-02*(x**6)+2.19600898e-01*(x**5)-4.96161512e-02*(x**4)-6.68335614e-01*(x**3)+1.31055577e-01*(x**2)+9.21971231e-01*x+4.55105144e-02),
                #  'custom_17': (lambda x: 1.62904037e-04*(x**9)+9.04216820e-04*(x**8)-2.91753357e-03*(x**7)-1.77029377e-02*(x**6)+1.80807173e-02*(x**5)+1.12807876e-01*(x**4)-3.46716429e-02*(x**3)-1.09476490e-01*(x**2)+4.10042084e-01*x+1.68902103e-01, lambda x: 1.62904037e-04*(x**9)+9.04216820e-04*(x**8)-2.91753357e-03*(x**7)-1.77029377e-02*(x**6)+1.80807173e-02*(x**5)+1.12807876e-01*(x**4)-3.46716429e-02*(x**3)-1.09476490e-01*(x**2)+4.10042084e-01*x+1.68902103e-01),
                 

                 'logcosh': (lambda x: torch.log(torch.cosh(x)), lambda x: sympy.log(sympy.cosh(x))),
                 'cosh^2': (lambda x: torch.cosh(x)**2, lambda x: sympy.cosh(x)**2),
}

def create_dataset(f, 
                   n_var=2, 
                   ranges = [-1,1],
                   train_num=1000, 
                   test_num=1000,
                   normalize_input=False,
                   normalize_label=False,
                   device='cpu',
                   seed=0):
    '''
    create dataset
    
    Args:
    -----
        f : function
            the symbolic formula used to create the synthetic dataset
        ranges : list or np.array; shape (2,) or (n_var, 2)
            the range of input variables. Default: [-1,1].
        train_num : int
            the number of training samples. Default: 1000.
        test_num : int
            the number of test samples. Default: 1000.
        normalize_input : bool
            If True, apply normalization to inputs. Default: False.
        normalize_label : bool
            If True, apply normalization to labels. Default: False.
        device : str
            device. Default: 'cpu'.
        seed : int
            random seed. Default: 0.
        
    Returns:
    --------
        dataset : dic
            Train/test inputs/labels are dataset['train_input'], dataset['train_label'],
                        dataset['test_input'], dataset['test_label']
         
    Example
    -------
    >>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
    >>> dataset = create_dataset(f, n_var=2, train_num=100)
    >>> dataset['train_input'].shape
    torch.Size([100, 2])
    '''

    np.random.seed(seed)
    torch.manual_seed(seed)

    if len(np.array(ranges).shape) == 1:
        ranges = np.array(ranges * n_var).reshape(n_var,2)
    else:
        ranges = np.array(ranges)
        
    train_input = torch.zeros(train_num, n_var)
    test_input = torch.zeros(test_num, n_var)
    for i in range(n_var):
        train_input[:,i] = torch.rand(train_num,)*(ranges[i,1]-ranges[i,0])+ranges[i,0]
        test_input[:,i] = torch.rand(test_num,)*(ranges[i,1]-ranges[i,0])+ranges[i,0]
        
        
    train_label = f(train_input)
    test_label = f(test_input)
        
        
    def normalize(data, mean, std):
            return (data-mean)/std
            
    if normalize_input == True:
        mean_input = torch.mean(train_input, dim=0, keepdim=True)
        std_input = torch.std(train_input, dim=0, keepdim=True)
        train_input = normalize(train_input, mean_input, std_input)
        test_input = normalize(test_input, mean_input, std_input)
        
    if normalize_label == True:
        mean_label = torch.mean(train_label, dim=0, keepdim=True)
        std_label = torch.std(train_label, dim=0, keepdim=True)
        train_label = normalize(train_label, mean_label, std_label)
        test_label = normalize(test_label, mean_label, std_label)

    dataset = {}
    dataset['train_input'] = train_input.to(device)
    dataset['test_input'] = test_input.to(device)

    dataset['train_label'] = train_label.to(device)
    dataset['test_label'] = test_label.to(device)

    return dataset



def fit_params(x, y, fun, a_range=(-10,10), b_range=(-10,10), grid_number=101, iteration=3, verbose=True, device='cpu'):
    '''
    fit a, b, c, d such that
    
    .. math::
        |y-(cf(ax+b)+d)|^2
        
    is minimized. Both x and y are 1D array. Sweep a and b, find the best fitted model.
    
    Args:
    -----
        x : 1D array
            x values
        y : 1D array
            y values
        fun : function
            symbolic function
        a_range : tuple
            sweeping range of a
        b_range : tuple
            sweeping range of b
        grid_num : int
            number of steps along a and b
        iteration : int
            number of zooming in
        verbose : bool
            print extra information if True
        device : str
            device
        
    Returns:
    --------
        a_best : float
            best fitted a
        b_best : float
            best fitted b
        c_best : float
            best fitted c
        d_best : float
            best fitted d
        r2_best : float
            best r2 (coefficient of determination)
    
    Example
    -------
    >>> num = 100
    >>> x = torch.linspace(-1,1,steps=num)
    >>> noises = torch.normal(0,1,(num,)) * 0.02
    >>> y = 5.0*torch.sin(3.0*x + 2.0) + 0.7 + noises
    >>> fit_params(x, y, torch.sin)
    r2 is 0.9999727010726929
    (tensor([2.9982, 1.9996, 5.0053, 0.7011]), tensor(1.0000))
    '''
    # fit a, b, c, d such that y=c*fun(a*x+b)+d; both x and y are 1D array.
    # sweep a and b, choose the best fitted model   
    for _ in range(iteration):
        a_ = torch.linspace(a_range[0], a_range[1], steps=grid_number, device=device)
        b_ = torch.linspace(b_range[0], b_range[1], steps=grid_number, device=device)
        a_grid, b_grid = torch.meshgrid(a_, b_, indexing='ij')
        post_fun = fun(a_grid[None,:,:] * x[:,None,None] + b_grid[None,:,:])
        x_mean = torch.mean(post_fun, dim=[0], keepdim=True)
        y_mean = torch.mean(y, dim=[0], keepdim=True)
        numerator = torch.sum((post_fun - x_mean)*(y-y_mean)[:,None,None], dim=0)**2
        denominator = torch.sum((post_fun - x_mean)**2, dim=0)*torch.sum((y - y_mean)[:,None,None]**2, dim=0)
        r2 = numerator/(denominator+1e-4)
        r2 = torch.nan_to_num(r2)
        
        
        best_id = torch.argmax(r2)
        a_id, b_id = torch.div(best_id, grid_number, rounding_mode='floor'), best_id % grid_number
        
        
        if a_id == 0 or a_id == grid_number - 1 or b_id == 0 or b_id == grid_number - 1:
            if _ == 0 and verbose==True:
                print('Best value at boundary.')
            if a_id == 0:
                a_range = [a_[0], a_[1]]
            if a_id == grid_number - 1:
                a_range = [a_[-2], a_[-1]]
            if b_id == 0:
                b_range = [b_[0], b_[1]]
            if b_id == grid_number - 1:
                b_range = [b_[-2], b_[-1]]
            
        else:
            a_range = [a_[a_id-1], a_[a_id+1]]
            b_range = [b_[b_id-1], b_[b_id+1]]
            
    a_best = a_[a_id]
    b_best = b_[b_id]
    post_fun = fun(a_best * x + b_best)
    r2_best = r2[a_id, b_id]
    
    if verbose == True:
        print(f"r2 is {r2_best}")
        if r2_best < 0.9:
            print(f'r2 is not very high, please double check if you are choosing the correct symbolic function.')

    post_fun = torch.nan_to_num(post_fun)
    reg = LinearRegression().fit(post_fun[:,None].detach().cpu().numpy(), y.detach().cpu().numpy())
    c_best = torch.from_numpy(reg.coef_)[0].to(device)
    d_best = torch.from_numpy(np.array(reg.intercept_)).to(device)
    return torch.stack([a_best, b_best, c_best, d_best]), r2_best



def add_symbolic(name, fun):
    '''
    add a symbolic function to library
    
    Args:
    -----
        name : str
            name of the function
        fun : fun
            torch function or lambda function
    
    Returns:
    --------
        None
    
    Example
    -------
    >>> print(SYMBOLIC_LIB['Bessel'])
    KeyError: 'Bessel'
    >>> add_symbolic('Bessel', torch.special.bessel_j0)
    >>> print(SYMBOLIC_LIB['Bessel'])
    (<built-in function special_bessel_j0>, Bessel)
    '''
    exec(f"globals()['{name}'] = sympy.Function('{name}')")
    SYMBOLIC_LIB[name] = (fun, globals()[name])
    
