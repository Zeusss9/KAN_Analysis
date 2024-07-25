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
                 
                'custom_40': (lambda x: torch.sinh(torch.tanh(0.24658921978683*torch.sin(torch.sin(1.39593438298758*x - 0.228283110289774))**2 - 0.02108932*torch.cosh(x + 2.3441336))), lambda x: sympy.sinh(sympy.tanh(0.24658921978683*sympy.sin(sympy.sin(1.39593438298758*x - 0.228283110289774))**2 - 0.02108932*sympy.cosh(x + 2.3441336)))),
                'custom_41': (lambda x: torch.tan(0.251296*torch.cos(torch.tanh(x))*torch.sinh(torch.cos(x - torch.cos(torch.atan(torch.sinh(torch.asinh(x)**2))**2)))), lambda x: sympy.tan(0.251296*sympy.cos(sympy.tanh(x))*sympy.sinh(sympy.cos(x - sympy.cos(sympy.atan(sympy.sinh(sympy.asinh(x)**2))**2))))),
                'custom_42': (lambda x: torch.sin(-0.109164505360561*torch.sin(x) + 0.3859853*torch.atan(x + 0.07101064)**3 + 0.020648947518216), lambda x: sympy.sin(-0.109164505360561*sympy.sin(x) + 0.3859853*sympy.atan(x + 0.07101064)**3 + 0.020648947518216)),
                'custom_43': (lambda x: -0.743232827939965*torch.asinh(-x + torch.sin(torch.sin(x) - 0.552021372215998) + 0.16371298), lambda x: -0.743232827939965*sympy.asinh(-x + sympy.sin(sympy.sin(x) - 0.552021372215998) + 0.16371298)),
                'custom_44': (lambda x: 0.31334865*torch.sin(-0.061096124*torch.cos(3.29950352370479*x) + 0.061096124*torch.sinh(x) + 0.061096124*torch.asinh(x)**4), lambda x: 0.31334865*sympy.sin(-0.061096124*sympy.cos(3.29950352370479*x) + 0.061096124*sympy.sinh(x) + 0.061096124*sympy.asinh(x)**4)),
                'custom_45': (lambda x: 0.137779107235427 - 0.081341766*torch.asinh(x**8 - x**3 + torch.asinh(22.3494137290616*x) + 1.9929446), lambda x: 0.137779107235427 - 0.081341766*sympy.asinh(x**8 - x**3 + sympy.asinh(22.3494137290616*x) + 1.9929446)),
                'custom_46': (lambda x: torch.sinh(torch.cos(0.91228604*x) - torch.asinh(torch.tanh(torch.atan(torch.exp(3*x))))), lambda x: sympy.sinh(sympy.cos(0.91228604*x) - sympy.asinh(sympy.tanh(sympy.atan(sympy.exp(3*x)))))),
                'custom_47': (lambda x: -torch.sinh(0.29449138*torch.sin(torch.sinh(torch.tan(torch.sin(x*(-0.20533957*x/torch.sqrt(x**2 + 1) + 0.68755394)))) + 0.339010575056295)), lambda x: -sympy.sinh(0.29449138*sympy.sin(sympy.sinh(sympy.tan(sympy.sin(x*(-0.20533957*x/sympy.sqrt(x**2 + 1) + 0.68755394)))) + 0.339010575056295))),
                'custom_48': (lambda x: 1.6656249710607*x*(torch.tanh(x + 0.5618634)**8 - 0.32731706)**2 + 0.0453519964870345, lambda x: 1.6656249710607*x*(sympy.tanh(x + 0.5618634)**8 - 0.32731706)**2 + 0.0453519964870345),
                'custom_49': (lambda x: torch.tan(torch.sin(torch.tanh(x)) - 0.72750247*torch.tanh(torch.sin(torch.asinh(x) + 0.3279781)**3) + 0.1831828) + 0.06531009, lambda x: sympy.tan(sympy.sin(sympy.tanh(x)) - 0.72750247*sympy.tanh(sympy.sin(sympy.asinh(x) + 0.3279781)**3) + 0.1831828) + 0.06531009),
                'custom_50': (lambda x: torch.sinh(0.26490757*torch.asinh(torch.sinh(torch.sinh(torch.sin(torch.sin(x + 1.21231780180099))) - 0.35719553) + torch.atan(x)**3)), lambda x: sympy.sinh(0.26490757*sympy.asinh(sympy.sinh(sympy.sinh(sympy.sin(sympy.sin(x + 1.21231780180099))) - 0.35719553) + sympy.atan(x)**3))),
                'custom_51': (lambda x: -torch.tanh(0.14937405*torch.cos(torch.asinh(1.11138620389405*torch.cosh(x)*torch.tanh(x**2))) + 0.14937405*torch.asinh(x**3)), lambda x: -sympy.tanh(0.14937405*sympy.cos(sympy.asinh(1.11138620389405*sympy.cosh(x)*sympy.tanh(x**2))) + 0.14937405*sympy.asinh(x**3))),
                'custom_52': (lambda x: -torch.atan(0.71872548696886*torch.tanh(torch.tan(torch.sqrt(x**2) - 3.29686481995693*torch.tanh(torch.sin(x**4)) + 0.11043203))), lambda x: -sympy.atan(0.71872548696886*sympy.tanh(sympy.tan(sympy.sqrt(x**2) - 3.29686481995693*sympy.tanh(sympy.sin(x**4)) + 0.11043203)))),
                'custom_53': (lambda x: torch.tanh(torch.tanh(0.41602978*torch.tanh(torch.sin(torch.tan(x**2) + 1.04993949636535/x)) - 0.04873254)), lambda x: sympy.tanh(sympy.tanh(0.41602978*sympy.tanh(sympy.sin(sympy.tan(x**2) + 1.04993949636535/x)) - 0.04873254))),
                'custom_54': (lambda x: torch.sinh(0.0718509*x/torch.sinh(x + torch.atan(torch.exp(torch.sin(torch.tan(x))))) + 0.0718509*torch.atan(0.6955437/torch.atan(x))), lambda x: sympy.sinh(0.0718509*x/sympy.sinh(x + sympy.atan(sympy.exp(sympy.sin(sympy.tan(x))))) + 0.0718509*sympy.atan(0.6955437/sympy.atan(x)))),
                'custom_55': (lambda x: torch.tanh(0.023664892*torch.sqrt(torch.exp(2*x) + 1)/torch.tan(torch.tan(torch.sin(torch.tan(torch.tan(torch.sin(2*x - 0.46984988))))))), lambda x: sympy.tanh(0.023664892*sympy.sqrt(sympy.exp(2*x) + 1)/sympy.tan(sympy.tan(sympy.sin(sympy.tan(sympy.tan(sympy.sin(2*x - 0.46984988)))))))),
                'custom_56': (lambda x: torch.tanh(torch.sinh(torch.sinh(torch.sin(torch.tan(x + 0.868080814852003))**9 - 0.1750511460305*torch.tan(x**3)))), lambda x: sympy.tanh(sympy.sinh(sympy.sinh(sympy.sin(sympy.tan(x + 0.868080814852003))**9 - 0.1750511460305*sympy.tan(x**3))))),
                'custom_57': (lambda x: torch.sinh(torch.atan(torch.asinh(torch.tan((x - 0.28283298)**8 - 0.399535965429853)**3 + torch.atan(0.975863*x)))), lambda x: sympy.sinh(sympy.atan(sympy.asinh(sympy.tan((x - 0.28283298)**8 - 0.399535965429853)**3 + sympy.atan(0.975863*x))))),
                'custom_58': (lambda x: -torch.atan(torch.tan(0.3616771*torch.sin(torch.sinh((-x - 0.06464093)**2) - 0.10345038)) - 0.08717201), lambda x: -sympy.atan(sympy.tan(0.3616771*sympy.sin(sympy.sinh((-x - 0.06464093)**2) - 0.10345038)) - 0.08717201)),
                'custom_59': (lambda x: -torch.sinh(0.15410076*torch.sqrt(torch.sinh(torch.tan(torch.cos(torch.tan(x + 0.18501645))**8))**3) + 0.15410076*torch.atan((x - 0.009419038)**3)), lambda x: -sympy.sinh(0.15410076*sympy.sqrt(sympy.sinh(sympy.tan(sympy.cos(sympy.tan(x + 0.18501645))**8))**3) + 0.15410076*sympy.atan((x - 0.009419038)**3))),
                'custom_60': (lambda x: -0.012068736 - torch.asinh(torch.sin(12.2873550337853*x**3 + 5.8276111120941)**3)/(2.277052 - x), lambda x: -0.012068736 - sympy.asinh(sympy.sin(12.2873550337853*x**3 + 5.8276111120941)**3)/(2.277052 - x)),
                'custom_61': (lambda x: -torch.tan(torch.tan(0.2564147*torch.tan(torch.sin(-torch.cos(0.10806315/x) + torch.sinh(x) + torch.atan(torch.asinh(x**3)))))), lambda x: -sympy.tan(sympy.tan(0.2564147*sympy.tan(sympy.sin(-sympy.cos(0.10806315/x) + sympy.sinh(x) + sympy.atan(sympy.asinh(x**3))))))),
                'custom_62': (lambda x: torch.atan(torch.atan(torch.atan(torch.atan(torch.atan(torch.atan(torch.sin(1.6505026*x + 0.95686435792712 - 0.78670144/x))))))), lambda x: sympy.atan(sympy.atan(sympy.atan(sympy.atan(sympy.atan(sympy.atan(sympy.sin(1.6505026*x + 0.95686435792712 - 0.78670144/x)))))))),
                'custom_63': (lambda x: -torch.atan(3.00101791526668*torch.tanh(torch.atan(torch.atan(torch.tan(torch.tan(torch.sin(x**2)) - torch.atan(x) + 0.015900327))/torch.sqrt(torch.atan(torch.tan(torch.tan(torch.sin(x**2)) - torch.atan(x) + 0.015900327))**2 + 1)))**3), lambda x: -sympy.atan(3.00101791526668*sympy.tanh(sympy.atan(sympy.atan(sympy.tan(sympy.tan(sympy.sin(x**2)) - sympy.atan(x) + 0.015900327))/sympy.sqrt(sympy.atan(sympy.tan(sympy.tan(sympy.sin(x**2)) - sympy.atan(x) + 0.015900327))**2 + 1)))**3)),
                'custom_64': (lambda x: torch.tan((0.18675241*torch.tanh(torch.tan(3.19464403870477*x)) - 0.143757074954376)*torch.tan(torch.asinh(torch.cos(x**2)) - 0.21643484)), lambda x: sympy.tan((0.18675241*sympy.tanh(sympy.tan(3.19464403870477*x)) - 0.143757074954376)*sympy.tan(sympy.asinh(sympy.cos(x**2)) - 0.21643484))),
                'custom_65': (lambda x: 1.0939739*torch.log(torch.atan(torch.sqrt(torch.cosh(torch.tan(x) + torch.atan((-torch.sin(x) + torch.sin(torch.cos(torch.sinh(x)**2)))**9)**2)))), lambda x: 1.0939739*sympy.log(sympy.atan(sympy.sqrt(sympy.cosh(sympy.tan(x) + sympy.atan((-sympy.sin(x) + sympy.sin(sympy.cos(sympy.sinh(x)**2)))**9)**2))))),
                'custom_66': (lambda x: torch.tan(0.369950448419946*torch.tanh(torch.sinh(torch.sin(torch.tan(0.8989355/x)) + 4.363231*torch.sin(torch.tan(x))))), lambda x: sympy.tan(0.369950448419946*sympy.tanh(sympy.sinh(sympy.sin(sympy.tan(0.8989355/x)) + 4.363231*sympy.sin(sympy.tan(x)))))),
                'custom_67': (lambda x: -torch.sinh(torch.sin(-torch.log(torch.sinh(x**2)) + torch.asinh(torch.cos(2.15852342837585*torch.asinh(x) - 1.5276578708786)) + 0.55881536)), lambda x: -sympy.sinh(sympy.sin(-sympy.log(sympy.sinh(x**2)) + sympy.asinh(sympy.cos(2.15852342837585*sympy.asinh(x) - 1.5276578708786)) + 0.55881536))),
                'custom_68': (lambda x: torch.tan(0.00260755499891598*(-0.437378914346524 - 1/x)**3 + torch.tanh(torch.sin((x + 0.09044661)**9))**2 - 0.014699431), lambda x: sympy.tan(0.00260755499891598*(-0.437378914346524 - 1/x)**3 + sympy.tanh(sympy.sin((x + 0.09044661)**9))**2 - 0.014699431)),
                'custom_69': (lambda x: torch.tan(torch.tan(torch.tanh(torch.sin(torch.sinh(torch.sin(x)/(-x + torch.atan(torch.cos(torch.tan(torch.asinh(x))))))) - 0.013589722))), lambda x: sympy.tan(sympy.tan(sympy.tanh(sympy.sin(sympy.sinh(sympy.sin(x)/(-x + sympy.atan(sympy.cos(sympy.tan(sympy.asinh(x))))))) - 0.013589722)))),
                'custom_70': (lambda x: torch.asinh(torch.atan((x + 0.09801989)**3 - torch.tanh(torch.atan(1.01346972203465*torch.tanh(x + 0.07214041))))) - 0.000993075156479025/x**2, lambda x: sympy.asinh(sympy.atan((x + 0.09801989)**3 - sympy.tanh(sympy.atan(1.01346972203465*sympy.tanh(x + 0.07214041))))) - 0.000993075156479025/x**2),
                'custom_71': (lambda x: torch.tan(torch.sinh(torch.log(1.0741748*torch.sinh(torch.cos(torch.sin(x - torch.tan(0.6474314/(x**2 + 1)**(3/2)))**3))))), lambda x: sympy.tan(sympy.sinh(sympy.log(1.0741748*sympy.sinh(sympy.cos(sympy.sin(x - sympy.tan(0.6474314/(x**2 + 1)**(3/2)))**3)))))),
                'custom_72': (lambda x: torch.asinh(0.92929125*torch.tan(0.8338829/torch.tan(torch.tan(torch.sinh(torch.atan(x))) - torch.cosh(torch.sin(torch.cos(1.40027201889481/x)**8))))), lambda x: sympy.asinh(0.92929125*sympy.tan(0.8338829/sympy.tan(sympy.tan(sympy.sinh(sympy.atan(x))) - sympy.cosh(sympy.sin(sympy.cos(1.40027201889481/x)**8)))))),
                'custom_73': (lambda x: torch.sin((-0.01462811*x + torch.tan(torch.sinh(torch.sin(1031.53567928692*(-0.462560364127519*x - 1)**9))))*(x - 0.00995389)), lambda x: sympy.sin((-0.01462811*x + sympy.tan(sympy.sinh(sympy.sin(1031.53567928692*(-0.462560364127519*x - 1)**9))))*(x - 0.00995389))),
                'custom_74': (lambda x: torch.tan(torch.tanh(torch.sin(3.57598718881134*torch.asinh(x) - 0.125951740050334))/torch.cosh(torch.tan(0.47972915*x) - 0.69755226)), lambda x: sympy.tan(sympy.tanh(sympy.sin(3.57598718881134*sympy.asinh(x) - 0.125951740050334))/sympy.cosh(sympy.tan(0.47972915*x) - 0.69755226))),
                'custom_75': (lambda x: torch.sin((x + torch.asinh(x**2 + torch.sin(torch.tan(x) - 0.17483409)))*torch.sin(torch.cos(x)**3)), lambda x: sympy.sin((x + sympy.asinh(x**2 + sympy.sin(sympy.tan(x) - 0.17483409)))*sympy.sin(sympy.cos(x)**3))),
                'custom_76': (lambda x: -torch.sin(torch.tan(torch.sin(1.03221373926033*torch.sin(2*x)*torch.sinh(torch.atan(x**3) + 0.9679062) + 0.10639457))), lambda x: -sympy.sin(sympy.tan(sympy.sin(1.03221373926033*sympy.sin(2*x)*sympy.sinh(sympy.atan(x**3) + 0.9679062) + 0.10639457)))),
                'custom_77': (lambda x: torch.tan(torch.tan(torch.sinh((torch.sin(3.14367440838405*x) + 0.11211546)/(-0.23098049*x**9 + 0.23098049*x + 3.176869)))), lambda x: sympy.tan(sympy.tan(sympy.sinh((sympy.sin(3.14367440838405*x) + 0.11211546)/(-0.23098049*x**9 + 0.23098049*x + 3.176869))))),
                'custom_78': (lambda x: torch.tan(torch.tan(torch.tanh(x + torch.cosh(x - torch.sin(1.07140981793029*x) + 0.7285176))**2 - 0.52652085)), lambda x: sympy.tan(sympy.tan(sympy.tanh(x + sympy.cosh(x - sympy.sin(1.07140981793029*x) + 0.7285176))**2 - 0.52652085))),
                'custom_79': (lambda x: -torch.sinh(torch.sin(2.79991009499452*x))*torch.tanh(torch.tanh(0.70097196*torch.atan(torch.asinh(x)) + 0.31887895)) + 0.050400935, lambda x: -sympy.sinh(sympy.sin(2.79991009499452*x))*sympy.tanh(sympy.tanh(0.70097196*sympy.atan(sympy.asinh(x)) + 0.31887895)) + 0.050400935),
                'custom_80': (lambda x: torch.tan(0.2651192*torch.cos(2.98423517597134*torch.sin(torch.tanh(torch.sin(x + 0.868091766582325) + 0.22559795)))), lambda x: sympy.tan(0.2651192*sympy.cos(2.98423517597134*sympy.sin(sympy.tanh(sympy.sin(x + 0.868091766582325) + 0.22559795))))),
                'custom_81': (lambda x: -torch.tanh(0.21124999/(torch.tan(x) + torch.tan(torch.cosh(torch.sin(torch.asinh(x)) - torch.cos(0.90250337288073*x**3)**9)))), lambda x: -sympy.tanh(0.21124999/(sympy.tan(x) + sympy.tan(sympy.cosh(sympy.sin(sympy.asinh(x)) - sympy.cos(0.90250337288073*x**3)**9))))),
                'custom_82': (lambda x: 1.086729*(torch.sin(2.47595356*torch.atan(torch.sinh(x))) - 0.19295965)*torch.atan(x + 1.0835382), lambda x: 1.086729*(sympy.sin(2.47595356*sympy.atan(sympy.sinh(x))) - 0.19295965)*sympy.atan(x + 1.0835382)),
                'custom_83': (lambda x: (torch.cos(x - torch.sinh(x**2 - 1.3703592)) - 0.14053842)/torch.asinh(torch.exp(-torch.atan(torch.sin(1.3703592/x) - 1.5543823))), lambda x: (sympy.cos(x - sympy.sinh(x**2 - 1.3703592)) - 0.14053842)/sympy.asinh(sympy.exp(-sympy.atan(sympy.sin(1.3703592/x) - 1.5543823)))),
                'custom_84': (lambda x: -0.32987493*torch.atan(torch.tanh(x) + torch.asinh(torch.atan(torch.tan(12.5892981837003*x**2))**9 + 0.050608512)), lambda x: -0.32987493*sympy.atan(sympy.tanh(x) + sympy.asinh(sympy.atan(sympy.tan(12.5892981837003*x**2))**9 + 0.050608512))),
                'custom_85': (lambda x: torch.asinh((torch.cos(x**9) - 0.5718951)*torch.tan(torch.sin(torch.tan(torch.tan(torch.tan(torch.exp(x) + 0.021652268)))))), lambda x: sympy.asinh((sympy.cos(x**9) - 0.5718951)*sympy.tan(sympy.sin(sympy.tan(sympy.tan(sympy.tan(sympy.exp(x) + 0.021652268))))))),
                'custom_86': (lambda x: -torch.tan(torch.asinh(torch.sin(x - 0.23628001))**8 - torch.asinh(torch.atan(x*torch.asinh(torch.log(torch.tanh(torch.atan(x**8))))))), lambda x: -sympy.tan(sympy.asinh(sympy.sin(x - 0.23628001))**8 - sympy.asinh(sympy.atan(x*sympy.asinh(sympy.log(sympy.tanh(sympy.atan(x**8)))))))),
                'custom_87': (lambda x: torch.sin(0.526015489998946*torch.tanh(torch.sin(1.28502826419667*x + torch.tanh(torch.tan(torch.sin(2*x))) - 0.9535711))), lambda x: sympy.sin(0.526015489998946*sympy.tanh(sympy.sin(1.28502826419667*x + sympy.tanh(sympy.tan(sympy.sin(2*x))) - 0.9535711)))),
                'custom_88': (lambda x: 0.54350718822436*torch.tan(torch.tan(torch.asinh(0.183213737194392/torch.atan(torch.atan(x**9))**3 - 0.889695928941505/x))), lambda x: 0.54350718822436*sympy.tan(sympy.tan(sympy.asinh(0.183213737194392/sympy.atan(sympy.atan(x**9))**3 - 0.889695928941505/x)))),
                'custom_89': (lambda x: torch.tanh(torch.tanh((torch.tan(0.7724108/torch.sinh(x)) + 3.759567)*(torch.atan(torch.sin(x**2 + 0.13486192))**9 - 0.015911901))), lambda x: sympy.tanh(sympy.tanh((sympy.tan(0.7724108/sympy.sinh(x)) + 3.759567)*(sympy.atan(sympy.sin(x**2 + 0.13486192))**9 - 0.015911901)))),
                'custom_90': (lambda x: torch.tan(0.21156946*torch.tan(torch.sin(torch.sin(x**6 + x - 1.0737734)) + 0.19357345)), lambda x: sympy.tan(0.21156946*sympy.tan(sympy.sin(sympy.sin(x**6 + x - 1.0737734)) + 0.19357345))),
                'custom_91': (lambda x: torch.sinh(torch.sin(torch.tan(torch.tanh(x**27 + x) - 0.5548052))/torch.sqrt(torch.cos(torch.sin(torch.sinh(x)))**2 + 1)), lambda x: sympy.sinh(sympy.sin(sympy.tan(sympy.tanh(x**27 + x) - 0.5548052))/sympy.sqrt(sympy.cos(sympy.sin(sympy.sinh(x)))**2 + 1))),
                'custom_92': (lambda x: torch.tanh(torch.tanh(torch.tanh(torch.cos(x)*torch.tan(torch.tan(torch.cos(torch.asinh(torch.cos(torch.asinh(torch.cos(x**2))) + 1.2128683/torch.tan(x)))))))), lambda x: sympy.tanh(sympy.tanh(sympy.tanh(sympy.cos(x)*sympy.tan(sympy.tan(sympy.cos(sympy.asinh(sympy.cos(sympy.asinh(sympy.cos(x**2))) + 1.2128683/sympy.tan(x))))))))),
                'custom_93': (lambda x: -1.02485452599945*torch.tan(0.5711505*torch.sinh(torch.sin(x))) + 1.02485452599945*torch.atan(torch.sinh(x + 0.18243542)), lambda x: -1.02485452599945*sympy.tan(0.5711505*sympy.sinh(sympy.sin(x))) + 1.02485452599945*sympy.atan(sympy.sinh(x + 0.18243542))),
                'custom_94': (lambda x: torch.sinh(torch.atan(0.16506639*x*(x - torch.sin(torch.cos(x)**2))) - 0.055672597), lambda x: sympy.sinh(sympy.atan(0.16506639*x*(x - sympy.sin(sympy.cos(x)**2))) - 0.055672597)),
                'custom_95': (lambda x: -torch.tanh(0.2262174*x*torch.sin(1.50523167351838*torch.tan(torch.tanh(x - 0.7745963))) - 0.11994322533228), lambda x: -sympy.tanh(0.2262174*x*sympy.sin(1.50523167351838*sympy.tan(sympy.tanh(x - 0.7745963))) - 0.11994322533228)),
                'custom_96': (lambda x: torch.tanh(torch.tanh(0.1145312*x - 0.15400118265952*torch.asinh(torch.sin(x + torch.asinh(x - 0.6614486))) + 0.025657501)), lambda x: sympy.tanh(sympy.tanh(0.1145312*x - 0.15400118265952*sympy.asinh(sympy.sin(x + sympy.asinh(x - 0.6614486))) + 0.025657501))),
                'custom_97': (lambda x: torch.tanh(torch.tanh(torch.tanh(torch.tanh(0.706560684448161*x)**16)) - 0.115541/torch.asinh(torch.tanh(x) + 2.6224716)), lambda x: sympy.tanh(sympy.tanh(sympy.tanh(sympy.tanh(0.706560684448161*x)**16)) - 0.115541/sympy.asinh(sympy.tanh(x) + 2.6224716))),
                'custom_98': (lambda x: (0.0857214625307289 - 0.231403570256264*torch.sin(torch.sinh(torch.atan(x - 0.28091088))**3))*torch.sinh(((x - 0.9286296)**2 + 1)**(-1/4)), lambda x: (0.0857214625307289 - 0.231403570256264*sympy.sin(sympy.sinh(sympy.atan(x - 0.28091088))**3))*sympy.sinh(((x - 0.9286296)**2 + 1)**(-1/4))),
                'custom_99': (lambda x: -torch.atan(torch.atan(torch.tanh(torch.cos(torch.sin(torch.cos(x - 0.15647557)))) - torch.asinh(0.047145024 + 1/torch.sqrt((x - 0.13281114)**2 + 1)))), lambda x: -sympy.atan(sympy.atan(sympy.tanh(sympy.cos(sympy.sin(sympy.cos(x - 0.15647557)))) - sympy.asinh(0.047145024 + 1/sympy.sqrt((x - 0.13281114)**2 + 1))))),
                'custom_100': (lambda x: torch.atan(x*torch.atan(torch.tanh(torch.cos(torch.tanh(torch.cos(x)) - 0.069739))) - torch.tanh(torch.sinh(x + 0.25829262)) + 0.20105115), lambda x: sympy.atan(x*sympy.atan(sympy.tanh(sympy.cos(sympy.tanh(sympy.cos(x)) - 0.069739))) - sympy.tanh(sympy.sinh(x + 0.25829262)) + 0.20105115)),
                'custom_101': (lambda x: torch.asinh(1.202*(torch.cos(torch.cos(torch.tanh(x)))**8 - 0.022868564)*(x + torch.sin(x)**2 - 0.45357233)), lambda x: sympy.asinh(1.202*(sympy.cos(sympy.cos(sympy.tanh(x)))**8 - 0.022868564)*(x + sympy.sin(x)**2 - 0.45357233))),
                'custom_102': (lambda x: torch.sin(0.980994891665301*torch.cos(torch.sinh(1.0193733*torch.asinh(x)) + 2.896043) + 0.980994891665301*torch.atan(x + 0.849733026089251) + 0.15974500214985), lambda x: sympy.sin(0.980994891665301*sympy.cos(sympy.sinh(1.0193733*sympy.asinh(x)) + 2.896043) + 0.980994891665301*sympy.atan(x + 0.849733026089251) + 0.15974500214985)),
                'custom_103': (lambda x: (torch.cos(torch.asinh(x + 0.1460395)/torch.sqrt(torch.asinh(x + 0.1460395)**2 + 1)) - 0.8678112)/torch.atan(torch.cos(torch.sin(1.7381104*x))**(1/4)), lambda x: (sympy.cos(sympy.asinh(x + 0.1460395)/sympy.sqrt(sympy.asinh(x + 0.1460395)**2 + 1)) - 0.8678112)/sympy.atan(sympy.cos(sympy.sin(1.7381104*x))**(1/4))),
                'custom_104': (lambda x: -torch.sinh(0.13932627*x + 0.13932627*torch.cos(x - torch.tanh(torch.sin(torch.sin(1.18511607974978*torch.sqrt(x**2)))))), lambda x: -sympy.sinh(0.13932627*x + 0.13932627*sympy.cos(x - sympy.tanh(sympy.sin(sympy.sin(1.18511607974978*sympy.sqrt(x**2))))))),
                'custom_105': (lambda x: torch.asinh(0.07909375*torch.sinh((torch.sin(2.28468187061527*torch.sin(0.41484228*x - 0.10327339)) + 0.06302366)**3)), lambda x: sympy.asinh(0.07909375*sympy.sinh((sympy.sin(2.28468187061527*sympy.sin(0.41484228*x - 0.10327339)) + 0.06302366)**3))),
                'custom_106': (lambda x: torch.sinh(torch.cos(x*(1.02064989065267*torch.tanh((torch.asinh(x) - 0.19717287)**3) - 1.13364614211182))) - 0.29240924, lambda x: sympy.sinh(sympy.cos(x*(1.02064989065267*sympy.tanh((sympy.asinh(x) - 0.19717287)**3) - 1.13364614211182))) - 0.29240924),
                'custom_107': (lambda x: -x + torch.cos(x + torch.sinh(torch.sinh(torch.cos(x))) + torch.tanh(torch.tanh(x))) - torch.tanh(torch.tanh(torch.sin(x)**8)), lambda x: -x + sympy.cos(x + sympy.sinh(sympy.sinh(sympy.cos(x))) + sympy.tanh(sympy.tanh(x))) - sympy.tanh(sympy.tanh(sympy.sin(x)**8))),
                'custom_108': (lambda x: torch.sinh(1.02681767444423*x*torch.asinh(torch.sin(torch.exp(torch.sin(torch.tanh(torch.sinh(1.06761273667546*x - 0.31739303))))))**2) - 0.03627557, lambda x: sympy.sinh(1.02681767444423*x*sympy.asinh(sympy.sin(sympy.exp(sympy.sin(sympy.tanh(sympy.sinh(1.06761273667546*x - 0.31739303))))))**2) - 0.03627557),
                'custom_109': (lambda x: torch.tan(torch.tan(torch.sin(torch.asinh(torch.sin(torch.tanh(0.861597*torch.sinh(torch.sin(x)) + 1.0058783)**2*torch.asinh(x)**2))) - 0.17874831)), lambda x: sympy.tan(sympy.tan(sympy.sin(sympy.asinh(sympy.sin(sympy.tanh(0.861597*sympy.sinh(sympy.sin(x)) + 1.0058783)**2*sympy.asinh(x)**2))) - 0.17874831))),
                'custom_110': (lambda x: torch.tan(torch.atan(torch.sinh(x + torch.cos((x + 0.43968734)*torch.sin(x)))*torch.atan(x + 0.15415446)) - 0.40805057), lambda x: sympy.tan(sympy.atan(sympy.sinh(x + sympy.cos((x + 0.43968734)*sympy.sin(x)))*sympy.atan(x + 0.15415446)) - 0.40805057)),
                'custom_111': (lambda x: torch.tanh(torch.sinh(0.28258723*x + torch.sqrt(torch.tanh(torch.atan((-x - 0.84974295)**2) - 2.1315966)**8)) - 0.57459116), lambda x: sympy.tanh(sympy.sinh(0.28258723*x + sympy.sqrt(sympy.tanh(sympy.atan((-x - 0.84974295)**2) - 2.1315966)**8)) - 0.57459116)),
                'custom_112': (lambda x: 2.08634693107859*torch.cos((x - 1.634175)*(torch.sin(x) + 0.31673753)) + torch.tan(torch.sin(0.5487091*x)) - 0.75594854, lambda x: 2.08634693107859*sympy.cos((x - 1.634175)*(sympy.sin(x) + 0.31673753)) + sympy.tan(sympy.sin(0.5487091*x)) - 0.75594854),
                'custom_113': (lambda x: 0.77301836*(torch.atan(x) - 0.5618751)*torch.cos(torch.sin(torch.sin(torch.cos(torch.cos(x)) - torch.atan(torch.sinh(x)))))**8, lambda x: 0.77301836*(sympy.atan(x) - 0.5618751)*sympy.cos(sympy.sin(sympy.sin(sympy.cos(sympy.cos(x)) - sympy.atan(sympy.sinh(x)))))**8),
                'custom_114': (lambda x: torch.tanh(x - torch.sin(torch.sin(1.99968644916477*x - 0.232385853898109) + 0.71725976) + 0.675446939281473) - 0.06907394, lambda x: sympy.tanh(x - sympy.sin(sympy.sin(1.99968644916477*x - 0.232385853898109) + 0.71725976) + 0.675446939281473) - 0.06907394),
                'custom_115': (lambda x: -torch.sinh(torch.sin(torch.atan(x - torch.cosh(torch.sin(x))**2) + 0.13060796)) - torch.asinh(x)**2, lambda x: -sympy.sinh(sympy.sin(sympy.atan(x - sympy.cosh(sympy.sin(x))**2) + 0.13060796)) - sympy.asinh(x)**2),
                'custom_116': (lambda x: 0.9142958*torch.atan(torch.atan(torch.sin(0.773627891637322*x + 0.773627891637322*torch.sin(torch.cos(1.20542620193951*x + torch.asinh(torch.exp(x))))))), lambda x: 0.9142958*sympy.atan(sympy.atan(sympy.sin(0.773627891637322*x + 0.773627891637322*sympy.sin(sympy.cos(1.20542620193951*x + sympy.asinh(sympy.exp(x)))))))),
                'custom_117': (lambda x: torch.tanh(0.420522756737108*torch.atan((torch.atan(x) + 0.340832419005196)/torch.cos(torch.cos(x*(-torch.sinh(torch.atan(x)) - 0.39190665))))), lambda x: sympy.tanh(0.420522756737108*sympy.atan((sympy.atan(x) + 0.340832419005196)/sympy.cos(sympy.cos(x*(-sympy.sinh(sympy.atan(x)) - 0.39190665)))))),
                'custom_118': (lambda x: -0.48180142*torch.sin(torch.sinh(torch.sinh(torch.sin(x + 0.33018526) + 0.48246366) - torch.tanh(x)) - 0.9855882), lambda x: -0.48180142*sympy.sin(sympy.sinh(sympy.sinh(sympy.sin(x + 0.33018526) + 0.48246366) - sympy.tanh(x)) - 0.9855882)),
                'custom_119': (lambda x: torch.sin(1.2243556*torch.log(torch.atan(torch.cosh(1.2463446*torch.cos(x + 0.06764326) + torch.sinh(x) - 0.86539538470728)))) - 0.007975344, lambda x: sympy.sin(1.2243556*sympy.log(sympy.atan(sympy.cosh(1.2463446*sympy.cos(x + 0.06764326) + sympy.sinh(x) - 0.86539538470728)))) - 0.007975344),
                'custom_120': (lambda x: -0.3190351*torch.sin((x - 0.37434092)/(0.5562881 + 0.31570917*torch.exp(-2*x))) - 0.06020562, lambda x: -0.3190351*sympy.sin((x - 0.37434092)/(0.5562881 + 0.31570917*sympy.exp(-2*x))) - 0.06020562),
                'custom_121': (lambda x: 0.059010245*x + 0.338142828961068*torch.sin(1.89375127413953*x - 0.7283189) + 0.0504203, lambda x: 0.059010245*x + 0.338142828961068*sympy.sin(1.89375127413953*x - 0.7283189) + 0.0504203),
                'custom_122': (lambda x: torch.sinh(0.41912428*torch.cos((-torch.asinh(torch.tan(torch.sin(x))) - 2.6790118)*torch.sinh(torch.sin(torch.asinh(x)))) - 0.032581724), lambda x: sympy.sinh(0.41912428*sympy.cos((-sympy.asinh(sympy.tan(sympy.sin(x))) - 2.6790118)*sympy.sinh(sympy.sin(sympy.asinh(x)))) - 0.032581724)),
                'custom_123': (lambda x: torch.sin(0.41044322*torch.sin(x + torch.tanh(x) + torch.atan(1.2214317*torch.sin(torch.cos(x - torch.tanh(x))**2)))), lambda x: sympy.sin(0.41044322*sympy.sin(x + sympy.tanh(x) + sympy.atan(1.2214317*sympy.sin(sympy.cos(x - sympy.tanh(x))**2))))),
                'custom_124': (lambda x: -torch.tan(0.30654153*torch.cos(1.71232337213184*x + torch.tan(torch.sin(torch.sin(1.04087061749233*x)) + 0.32824415) - 0.7459906) + 0.07392867), lambda x: -sympy.tan(0.30654153*sympy.cos(1.71232337213184*x + sympy.tan(sympy.sin(sympy.sin(1.04087061749233*x)) + 0.32824415) - 0.7459906) + 0.07392867)),
                'custom_125': (lambda x: torch.tan(torch.tan(0.65404433*torch.sqrt(torch.cos(0.9314263*x)**10) + 0.506243588519899*torch.sin(x - 0.31467122))), lambda x: sympy.tan(sympy.tan(0.65404433*sympy.sqrt(sympy.cos(0.9314263*x)**10) + 0.506243588519899*sympy.sin(x - 0.31467122)))),
                'custom_126': (lambda x: torch.tan(torch.atan(torch.sin(x + 0.31319332))/(-0.887561648922682*x**3 + torch.sin(torch.sinh((x + 0.2777891)**3)) + 4.5171685)), lambda x: sympy.tan(sympy.atan(sympy.sin(x + 0.31319332))/(-0.887561648922682*x**3 + sympy.sin(sympy.sinh((x + 0.2777891)**3)) + 4.5171685))),
                'custom_127': (lambda x: torch.tanh(0.23600331*torch.tanh(x) + 0.23600331*torch.tanh((torch.tan(1.7573681*x)**2 - 0.83824354)/torch.asinh(torch.cosh(x)))), lambda x: sympy.tanh(0.23600331*sympy.tanh(x) + 0.23600331*sympy.tanh((sympy.tan(1.7573681*x)**2 - 0.83824354)/sympy.asinh(sympy.cosh(x))))),
                'custom_128': (lambda x: torch.tanh(torch.tan(torch.sqrt(torch.atan((x + torch.sin(torch.tan(torch.sin(x)))**8)**2))) - 0.79753697) + torch.atan(torch.sin(torch.sin(x))), lambda x: sympy.tanh(sympy.tan(sympy.sqrt(sympy.atan((x + sympy.sin(sympy.tan(sympy.sin(x)))**8)**2))) - 0.79753697) + sympy.atan(sympy.sin(sympy.sin(x)))),
                'custom_129': (lambda x: torch.sqrt((torch.sin(torch.sin(x)) + 0.42304766)**10 + torch.cos(torch.tan(torch.sin(torch.sin(2.11773227637033*x + 3.92152217852305))))) - 1.1295476, lambda x: sympy.sqrt((sympy.sin(sympy.sin(x)) + 0.42304766)**10 + sympy.cos(sympy.tan(sympy.sin(sympy.sin(2.11773227637033*x + 3.92152217852305))))) - 1.1295476),
                'custom_130': (lambda x: torch.tanh(0.56491756*torch.tanh(torch.sinh(torch.sin(x)) + torch.tanh(torch.tanh(torch.sinh(torch.sinh(x))))**2)), lambda x: sympy.tanh(0.56491756*sympy.tanh(sympy.sinh(sympy.sin(x)) + sympy.tanh(sympy.tanh(sympy.sinh(sympy.sinh(x))))**2))),
                'custom_131': (lambda x: torch.sin(torch.sin(1.2239518*torch.sinh(x + 0.403983))**3 + torch.sinh(0.258578421373478/torch.asinh(torch.tanh(x)))**3), lambda x: sympy.sin(sympy.sin(1.2239518*sympy.sinh(x + 0.403983))**3 + sympy.sinh(0.258578421373478/sympy.asinh(sympy.tanh(x)))**3)),
                'custom_132': (lambda x: -torch.asinh(torch.sqrt(torch.sin(x + 0.115032166)**8) + 0.3316155*torch.sin(x) - 0.3316155*torch.tanh(torch.cosh(x + 0.119616225)**9)**2), lambda x: -sympy.asinh(sympy.sqrt(sympy.sin(x + 0.115032166)**8) + 0.3316155*sympy.sin(x) - 0.3316155*sympy.tanh(sympy.cosh(x + 0.119616225)**9)**2)),
                'custom_133': (lambda x: torch.sin(torch.cos((x - 0.037287425)*torch.tan(x + 0.28772232)))**11 + 0.003476185/torch.sin(x**10 - x), lambda x: sympy.sin(sympy.cos((x - 0.037287425)*sympy.tan(x + 0.28772232)))**11 + 0.003476185/sympy.sin(x**10 - x)),
                'custom_134': (lambda x: -0.5376953*torch.sin(-torch.sin(torch.tanh(1.51132453436836*(0.813431775931432*x + 1)**2)) + torch.atan(torch.asinh(x) + 1.4094014)**9 + 0.31776237), lambda x: -0.5376953*sympy.sin(-sympy.sin(sympy.tanh(1.51132453436836*(0.813431775931432*x + 1)**2)) + sympy.atan(sympy.asinh(x) + 1.4094014)**9 + 0.31776237)),
                'custom_135': (lambda x: torch.tan((torch.tanh(x**2) - 0.48091218)*torch.atan(torch.sin(x) + torch.cos(torch.asinh(22.0251464038699*torch.sinh(x)**8))**2)), lambda x: sympy.tan((sympy.tanh(x**2) - 0.48091218)*sympy.atan(sympy.sin(x) + sympy.cos(sympy.asinh(22.0251464038699*sympy.sinh(x)**8))**2))),
                'custom_136': (lambda x: torch.sinh(0.12897603*torch.sinh(x - torch.sinh(torch.tan(torch.cos(x)))) + torch.asinh(torch.cos(x + 0.81355417)**3)), lambda x: sympy.sinh(0.12897603*sympy.sinh(x - sympy.sinh(sympy.tan(sympy.cos(x)))) + sympy.asinh(sympy.cos(x + 0.81355417)**3))),
                'custom_137': (lambda x: torch.tan(0.40337726*torch.tan((torch.sin(x) + 0.44600925)*torch.cos(torch.sin(torch.sin(x**2)) + 0.51596147))), lambda x: sympy.tan(0.40337726*sympy.tan((sympy.sin(x) + 0.44600925)*sympy.cos(sympy.sin(sympy.sin(x**2)) + 0.51596147)))),
                'custom_138': (lambda x: -0.124621265740916*torch.sin(torch.tan(0.526926858231277*x**2)) - 0.5770604*torch.tan(38.4256869930379*(0.717700893989765*x + 1)**11), lambda x: -0.124621265740916*sympy.sin(sympy.tan(0.526926858231277*x**2)) - 0.5770604*sympy.tan(38.4256869930379*(0.717700893989765*x + 1)**11)),
                'custom_139': (lambda x: torch.tanh((torch.sin(torch.asinh(x + torch.cos(0.91665715*x))**2) - 0.021999137)*torch.atan(torch.tanh(torch.tanh(x)) - 0.1373645)), lambda x: sympy.tanh((sympy.sin(sympy.asinh(x + sympy.cos(0.91665715*x))**2) - 0.021999137)*sympy.atan(sympy.tanh(sympy.tanh(x)) - 0.1373645))),
                'custom_140': (lambda x: 0.060435947 - 0.69837016*torch.sin(-4.10452400225076*x + 4.10452400225076*torch.tanh(1.26790085826745*torch.cos(x - 0.111174144)) + 0.434383022933494), lambda x: 0.060435947 - 0.69837016*sympy.sin(-4.10452400225076*x + 4.10452400225076*sympy.tanh(1.26790085826745*sympy.cos(x - 0.111174144)) + 0.434383022933494)),

                'custom_141':(
                    lambda x: (torch.sin(1.92417074013612*x + 1.788666) - 1.4447228*torch.tanh(x))/(0.1354819*x + 0.6339829),
                    lambda x: (sympy.sin(1.92417074013612*x + 1.788666) - 1.4447228*sympy.tanh(x))/(0.1354819*x + 0.6339829)
                ),
                'custom_142':(
                    lambda x: 0.3498128 + (9.935232*torch.sin(1.92559575044125*torch.sin(x + 0.1402125) - 1.11670352256933) + torch.tanh(x))/(x + 3.782789),
                    lambda x: 0.3498128 + (9.935232*sympy.sin(1.92559575044125*sympy.sin(x + 0.1402125) - 1.11670352256933) + sympy.tanh(x))/(x + 3.782789)
                ),
                'custom_143':(
                    lambda x: -torch.tanh(x - 2.0400648) - 0.97658795 - torch.tanh(9.750686/x)/(-0.6245924 - 6.2144313/x),
                    lambda x: -sympy.tanh(x - 2.0400648) - 0.97658795 - sympy.tanh(9.750686/x)/(-0.6245924 - 6.2144313/x)
                ), 
                'custom_144':(
                    lambda x: 2.5966544*torch.tanh(0.6901091*x - 1.1834452) - torch.tanh(x - torch.tanh(x + 0.7025417)) + 1.59524138770256,
                    lambda x: 2.5966544*sympy.tanh(0.6901091*x - 1.1834452) - sympy.tanh(x - sympy.tanh(x + 0.7025417)) + 1.59524138770256
                ),
                'custom_145':(
                    lambda x: (torch.cos(1.35925362948244*x - 1.13415416032127) + 0.6064735)*torch.sin(torch.asinh(torch.asinh(x))) - 0.6630668*torch.asinh(torch.exp(x)),
                    lambda x: (sympy.cos(1.35925362948244*x - 1.13415416032127) + 0.6064735)*sympy.sin(sympy.asinh(sympy.asinh(x))) - 0.6630668*sympy.asinh(sympy.exp(x))
                ),
                'custom_146':(
                    lambda x: -torch.exp(0.89393437/(-0.0011982681 - 0.36495507/(x + 1.0553331))) + 0.143796050399609*torch.exp(torch.asinh(x)),
                    lambda x: -sympy.exp(0.89393437/(-0.0011982681 - 0.36495507/(x + 1.0553331))) + 0.143796050399609*sympy.exp(sympy.asinh(x))
                ),
                'custom_147':(
                    lambda x: 6.449689*torch.exp(x/(-x - 3.8647997)) + 6.449689*torch.sin(0.20307373*x) - 7.3339615,
                    lambda x: 6.449689*sympy.exp(x/(-x - 3.8647997)) + 6.449689*sympy.sin(0.20307373*x) - 7.3339615
                ),
                'custom_148':(
                    lambda x: -2.4470899864182e-6*torch.exp(4.333077*x) + torch.cosh(0.250647654741277*torch.exp(torch.exp(-torch.sin(0.7286534*x)))),
                    lambda x: -2.4470899864182e-6*sympy.exp(4.333077*x) + sympy.cosh(0.250647654741277*sympy.exp(sympy.exp(-sympy.sin(0.7286534*x))))
                ),
                'custom_149':(
                    lambda x: 5.70326351770762e-6*torch.exp(4.119639*x) - torch.exp(-3.1347487*torch.sin(torch.asinh(3.5568726*x + 4.4037052776621))),
                    lambda x: 5.70326351770762e-6*sympy.exp(4.119639*x) - sympy.exp(-3.1347487*sympy.sin(sympy.asinh(3.5568726*x + 4.4037052776621)))
                ),
                'custom_151':(
                    lambda x: -0.0007564828*torch.exp(1.67648486054538*x) + torch.atan(5.969991/x),
                    lambda x: -0.0007564828*sympy.exp(1.67648486054538*x) + sympy.atan(5.969991/x)
                ),
                'custom_152':(
                    lambda x: 0.0009518649*torch.exp(1.6377283*x) + 3.490736*torch.sin(2.5877905*torch.tanh(x))/(-x - 3.1246195),
                    lambda x: 0.0009518649*sympy.exp(1.6377283*x) + 3.490736*sympy.sin(2.5877905*sympy.tanh(x))/(-x - 3.1246195)
                ),
                'custom_153':(
                    lambda x: (0.31140703*x*torch.tanh(torch.tanh(0.19384687*x) + 1.0002766) + 0.0115986649937519)*torch.tanh(x - 1.2041519),
                    lambda x: (0.31140703*x*sympy.tanh(sympy.tanh(0.19384687*x) + 1.0002766) + 0.0115986649937519)*sympy.tanh(x - 1.2041519)
                ),
                'custom_154':(
                    lambda x: torch.asinh((0.292845020472025 - 0.19123475*x)*torch.atan((0.14313224*x + torch.tanh(torch.sin(x)))*torch.exp(x))),
                    lambda x: sympy.asinh((0.292845020472025 - 0.19123475*x)*sympy.atan((0.14313224*x + sympy.tanh(sympy.sin(x)))*sympy.exp(x)))
                ),
                'custom_155':(
                    lambda x: 0.16296057 - (torch.atan(x + 1.9097139) - 0.98715186)*torch.sin(x + 1.0805099)/(0.05280678*x + 0.20382920696268),
                    lambda x: 0.16296057 - (sympy.atan(x + 1.9097139) - 0.98715186)*sympy.sin(x + 1.0805099)/(0.05280678*x + 0.20382920696268)
                ),
                'custom_156':(
                    lambda x: torch.cos(1.6044266*torch.atan(x) - 1.32241550968498) + torch.exp(-torch.atan(x + 1.2021043)/(0.0101485669005708*x + 0.329626335855861)),
                    lambda x: sympy.cos(1.6044266*sympy.atan(x) - 1.32241550968498) + sympy.exp(-sympy.atan(x + 1.2021043)/(0.0101485669005708*x + 0.329626335855861))
                ),
                'custom_157':(
                    lambda x: torch.exp(torch.asinh(x + 0.14632547))*torch.sin(x + 1.66382596114234*torch.sin(x) + 0.33284742),
                    lambda x: sympy.exp(sympy.asinh(x + 0.14632547))*sympy.sin(x + 1.66382596114234*sympy.sin(x) + 0.33284742)
                ),
                'custom_158':(
                    lambda x: (-1.93985544158452*torch.sin(2*x) + torch.atan(torch.exp(x) - 1.7297496))*torch.tanh(torch.exp(x)/x**2),
                    lambda x: (-1.93985544158452*sympy.sin(2*x) + sympy.atan(sympy.exp(x) - 1.7297496))*sympy.tanh(sympy.exp(x)/x**2)
                ),
                'custom_159':(
                    lambda x: torch.tanh(torch.sin(0.3490692*x) + torch.asinh(torch.cos(x) - 0.7664667))/torch.sin(torch.exp(-torch.asinh(torch.asinh(x) + 0.13614361))),
                    lambda x: sympy.tanh(sympy.sin(0.3490692*x) + sympy.asinh(sympy.cos(x) - 0.7664667))/sympy.sin(sympy.exp(-sympy.asinh(sympy.asinh(x) + 0.13614361)))
                ),
                'custom_160':(
                    lambda x: 3.493945*(torch.cos(1.7954202*x - 1.7954202*torch.tanh(torch.tanh(x + 0.878547))) - 0.22824745)/(-x - 2.367978),
                    lambda x: 3.493945*(sympy.cos(1.7954202*x - 1.7954202*sympy.tanh(sympy.tanh(x + 0.878547))) - 0.22824745)/(-x - 2.367978)
                ),
                'custom_161':(
                    lambda x: -torch.asinh(torch.sin(1.8613503*x + 1.2111188)/torch.sqrt(x**2*torch.asinh(x - 1.5921078)**2 + 1)),
                    lambda x: -sympy.asinh(sympy.sin(1.8613503*x + 1.2111188)/sympy.sqrt(x**2*sympy.asinh(x - 1.5921078)**2 + 1))
                ),
                'custom_162':(
                    lambda x: torch.asinh(torch.sin(19.9767880092642*torch.exp(0.0916313290446206*x))*torch.cos(x*torch.asinh(torch.cos(torch.tanh(x + 0.93207043))))),
                    lambda x: sympy.asinh(sympy.sin(19.9767880092642*sympy.exp(0.0916313290446206*x))*sympy.cos(x*sympy.asinh(sympy.cos(sympy.tanh(x + 0.93207043)))))
                ),
                'custom_163':(
                    lambda x: (x - torch.asinh(torch.exp(torch.cos(2.02742949918031*x))))*torch.tanh(torch.exp(x + x/(x + 1.2613581))),
                    lambda x: (x - sympy.asinh(sympy.exp(sympy.cos(2.02742949918031*x))))*sympy.tanh(sympy.exp(x + x/(x + 1.2613581)))
                ),
                'custom_164':(
                    lambda x: torch.asinh(torch.exp(x) - torch.exp(torch.sin(x + torch.tanh(x + torch.cos(x)))))/(0.433708896323623*x + 0.758510192592773),
                    lambda x: sympy.asinh(sympy.exp(x) - sympy.exp(sympy.sin(x + sympy.tanh(x + sympy.cos(x)))))/(0.433708896323623*x + 0.758510192592773),
                ),
                'custom_165':(
                    lambda x: torch.sin(4.00650367728927*torch.atan(x - 0.55267936))/torch.tanh(torch.tanh(torch.cos(x - torch.sin(x - 1.2160506)))),
                    lambda x: sympy.sin(4.00650367728927*sympy.atan(x - 0.55267936))/sympy.tanh(sympy.tanh(sympy.cos(x - sympy.sin(x - 1.2160506))))
                ),
                'custom_166':(
                    lambda x: 3.316147*torch.cos(6.054305*torch.exp(torch.atan(0.533279550313086*torch.asinh(x)))) - 0.24120553/(x + 1.741058),
                    lambda x: 3.316147*sympy.cos(6.054305*sympy.exp(sympy.atan(0.533279550313086*sympy.asinh(x)))) - 0.24120553/(x + 1.741058)
                ),
                'custom_167':(
                    lambda x: -8.642741*(1.4114401 - x)*torch.sin(0.452214524507581*torch.exp(x)*torch.tanh(torch.sin(x))),
                    lambda x: -8.642741*(1.4114401 - x)*sympy.sin(0.452214524507581*sympy.exp(x)*sympy.tanh(sympy.sin(x)))
                ),
                'custom_168':(
                    lambda x: torch.sin(2.7166667*x) + torch.tanh(7.1720753 - 11.176988/x)/torch.sin(0.5705867*x - 1.7768629012966),
                    lambda x: sympy.sin(2.7166667*x) + sympy.tanh(7.1720753 - 11.176988/x)/sympy.sin(0.5705867*x - 1.7768629012966)
                ),
                'custom_169':(
                    lambda x: torch.exp(-0.679868442736857*torch.cos(x - 0.3876641) + torch.asinh(x - 3.6090524)),
                    lambda x: sympy.exp(-0.679868442736857*sympy.cos(x - 0.3876641) + sympy.asinh(x - 3.6090524))
                ),
                'custom_170':(
                    lambda x: -torch.sin(6.24654487986333*torch.tanh(torch.exp(x))) + 15.015239*torch.atan(x - 10.954447) - 15.015239*torch.atan(x - 9.0260315),
                    lambda x: -sympy.sin(6.24654487986333*sympy.tanh(sympy.exp(x))) + 15.015239*sympy.atan(x - 10.954447) - 15.015239*sympy.atan(x - 9.0260315),
                ),
                'custom_171':(
                    lambda x: -0.17998649*x - 2.5057697*torch.sin(2.1850028*torch.sin(x + 0.45795035) - 0.950373916168904),
                    lambda x: -0.17998649*x - 2.5057697*sympy.sin(2.1850028*sympy.sin(x + 0.45795035) - 0.950373916168904)
                ),



                # 'custom_141':(
                #     lambda x: (torch.sin(1.92417074013612*x + 1.788666) - 1.4447228*torch.tanh(x))/(0.1354819*x + 0.6339829),
                #     lambda x: (sympy.sin(1.92417074013612*x + 1.788666) - 1.4447228*sympy.tanh(x))/(0.1354819*x + 0.6339829)
                # ),
                # 'custom_142':(
                #     lambda x: 0.3498128 + (9.935232*torch.sin(1.92559575044125*torch.sin(x + 0.1402125) - 1.11670352256933) + torch.tanh(x))/(x + 3.782789),
                #     lambda x: 0.3498128 + (9.935232*sympy.sin(1.92559575044125*sympy.sin(x + 0.1402125) - 1.11670352256933) + sympy.tanh(x))/(x + 3.782789)
                # ),
                # 'custom_143':(
                #     lambda x: -torch.tanh(x - 2.0400648) - 0.97658795 - torch.tanh(9.750686/x)/(-0.6245924 - 6.2144313/x),
                #     lambda x: -sympy.tanh(x - 2.0400648) - 0.97658795 - sympy.tanh(9.750686/x)/(-0.6245924 - 6.2144313/x)
                # ), 
                # 'custom_144':(
                #     lambda x: 2.5966544*torch.tanh(0.6901091*x - 1.1834452) - torch.tanh(x - torch.tanh(x + 0.7025417)) + 1.59524138770256,
                #     lambda x: 2.5966544*sympy.tanh(0.6901091*x - 1.1834452) - sympy.tanh(x - sympy.tanh(x + 0.7025417)) + 1.59524138770256
                # ),
                # 'custom_145':(
                #     lambda x: (-0.6620355*x*(torch.exp(10*x + 100) + 1)*torch.exp(10*x - 100) + ((1.1618576*x - 0.497085646774048)*torch.sin(1.3623214*torch.sin(x + 0.3873579) - torch.atan(x)) - 0.167689722313248)*torch.exp(10*x + 100))/((torch.exp(10*x - 100) + 1)*(torch.exp(10*x + 100) + 1)),
                #     lambda x: (-0.6620355*x*(sympy.exp(10*x + 100) + 1)*sympy.exp(10*x - 100) + ((1.1618576*x - 0.497085646774048)*sympy.sin(1.3623214*sympy.sin(x + 0.3873579) - sympy.atan(x)) - 0.167689722313248)*sympy.exp(10*x + 100))/((sympy.exp(10*x - 100) + 1)*(sympy.exp(10*x + 100) + 1))
                # ),
                # 'custom_146':(
                #     lambda x: (x < -10).float() * torch.zeros_like(x) + ((x >= -10) & (x <= 10)).float() * (51.9513878899523*(0.644730319300757*torch.sin(0.3801634*x + 1.00676369431796) - 1)**9 + torch.sin(2.50877148044637*x - 0.739856679404619) + 0.54830927) + (x > 10).float() * (0.288082612238201*x),
                #     lambda x: 0 * sympy.And(x < -10) + (51.9513878899523*(0.644730319300757*sympy.sin(0.3801634*x + 1.00676369431796) - 1)**9 + sympy.sin(2.50877148044637*x - 0.739856679404619) + 0.54830927) * sympy.And(x >= -10, x <= 10) + (0.288082612238201*x) * sympy.And(x > 10)
                # ),
                # 'custom_147':(
                #     lambda x: (x < -200).float() * torch.zeros_like(x) + ((x >= -200) & (x <= 10)).float() * (-torch.sin(x*(torch.sin(x) + 0.8326211)) + torch.exp(-7.405453*torch.sin(torch.asinh(x) + 1.1280764))) + (x > 10).float() * (-0.092607185*x),
                #     lambda x: 0 * sympy.And(x < -200) + (-sympy.sin(x*(sympy.sin(x) + 0.8326211)) + sympy.exp(-7.405453*sympy.sin(sympy.asinh(x) + 1.1280764))) * sympy.And(x >= -200, x <= 10) + (-0.092607185*x) * sympy.And(x > 10)
                # ),
                # 'custom_148':(
                #     lambda x: (x < -10).float() * torch.zeros_like(x) + (x >= -10).float() * ((x + 1.2887747)*(torch.sin(torch.sin(x)) - 0.673435)**9 - torch.atan(38.0709491365401*x**11) + 1.5454899),
                #     lambda x: 0 * sympy.And(x < -10) + ((x + 1.2887747)*(sympy.sin(sympy.sin(x)) - 0.673435)**9 - sympy.atan(38.0709491365401*x**11) + 1.5454899) * sympy.And(x >= -10)
                # ),
                # 'custom_149':(
                #     lambda x: (x < -5).float() * torch.zeros_like(x) + (x >= -5).float() * (x + 1.2432121)*(torch.asinh(torch.sin(x)) - 0.70494455)**8 - torch.cos(2.37658557471495*x) + torch.tanh(x) - 1.3147533,
                #     lambda x: 0 * sympy.And(x < -5) + (x + 1.2432121)*(sympy.asinh(sympy.sin(x)) - 0.70494455)**8 - sympy.cos(2.37658557471495*x) + sympy.tanh(x) - 1.3147533 * sympy.And(x >= -5)
                # ),
                # 'custom_151':(
                #     lambda x: (x < -50).float() * torch.zeros_like(x) + (x >= -50).float() * (-8.19227626152316e-5*(0.724470527959346*x - 1)**10 + torch.sin(2.06217158628631*x) + torch.sin(x - 0.2726203) + 0.32098338),
                #     lambda x: 0 * sympy.And(x < -50) + (-8.19227626152316e-5*(0.724470527959346*x - 1)**10 + sympy.sin(2.06217158628631*x) + sympy.sin(x - 0.2726203) + 0.32098338) * sympy.And(x >= -50)
                # ),
                # 'custom_152':(
                #     lambda x: (x < -50).float() * torch.zeros_like(x) + (x >= -50).float() * (0.00145503867552248*(0.706639571278946*x - 1)**8 - torch.sin(2.0351255*x) - torch.tanh(torch.sin(x)*torch.atan(x + 1.7585657))),
                #     lambda x: 0 * sympy.And(x < -50) + (0.00145503867552248*(0.706639571278946*x - 1)**8 - sympy.sin(2.0351255*x) - sympy.tanh(sympy.sin(x)*sympy.atan(x + 1.7585657))) * sympy.And(x >= -50)
                # ),
                # 'custom_153':(
                #     lambda x: (x < -20).float() * torch.zeros_like(x) + ((x >= -20) & (x <= 8)).float() * (torch.tan(torch.tan(torch.sin(torch.sin(torch.tan(0.27664214*x + 0.0154092659715112))*torch.atan(torch.tanh(x - 1.206411))))) + (x > 8).float() * (0.3004957*x - 2.573844*torch.exp(-x))),
                #     lambda x: 0 * sympy.And(x < -20) + (sympy.tan(sympy.tan(sympy.sin(sympy.sin(sympy.tan(0.27664214*x + 0.0154092659715112))*sympy.atan(sympy.tanh(x - 1.206411)))))) * sympy.And(x >= -20, x <= 8) + (0.3004957*x - 2.573844*sympy.exp(-x)) * sympy.And(x > 8)
                # ),
                # 'custom_154':( 
                #     lambda x: (x < -120).float() * torch.zeros_like(x) + ((x >= -120) & (x <= 8)).float() * torch.tan(0.53190446*torch.sin(torch.tan(torch.cos(0.636181*x) - 0.50759023)*torch.tanh(x**8 + torch.tan(x)))) + (x > 8).float() * (-0.15891315*x + 1.38558071656444*torch.exp(-x)),
                #     lambda x: 0 * sympy.And(x < -120) + sympy.tan(0.53190446*sympy.sin(sympy.tan(sympy.cos(0.636181*x) - 0.50759023)*sympy.tanh(x**8 + sympy.tan(x)))) * sympy.And(x >= -120, x <= 8) + (-0.15891315*x + 1.38558071656444*sympy.exp(-x)) * sympy.And(x > 8)
                # ),
                # 'custom_155':(  
                #     lambda x: (x < -100).float() * torch.zeros_like(x) + (x >= -100).float() * (torch.atan(x + 0.53939354) - 0.31071404)**9 - torch.sin(x + 0.32633963)/torch.cosh(x),
                #     lambda x: 0 * sympy.And(x < -100) + (sympy.atan(x + 0.53939354) - 0.31071404)**9 - sympy.sin(x + 0.32633963)/sympy.cosh(x) * sympy.And(x >= -100)
                # ),
                # 'custom_156':(
                #     lambda x: (x < -10).float() * torch.zeros_like(x) + (x >= -10).float() * (torch.sin(torch.asinh(1.4513612*x + 0.286566639097772)) + torch.exp(-3.360346*torch.atan(x + 1.2375063))),
                #     lambda x: 0 * sympy.And(x < -10) + (sympy.sin(sympy.asinh(1.4513612*x + 0.286566639097772)) + sympy.exp(-3.360346*sympy.atan(x + 1.2375063))) * sympy.And(x >= -10)
                # ),
                # 'custom_157':( 
                #     lambda x: (x < -10).float() * torch.zeros_like(x) + (x >= -10).float() * 3.9079692*torch.cos(1.2520525*x)*torch.tanh(torch.sin(1.1001147*x - 0.6091945) + 0.70820284),
                #     lambda x: 0 * sympy.And(x < -10) + 3.9079692*sympy.cos(1.2520525*x)*sympy.tanh(sympy.sin(1.1001147*x - 0.6091945) + 0.70820284) * sympy.And(x >= -10)
                # ),
                # 'custom_158':(   
                #     lambda x: (x < -2).float() * torch.zeros_like(x) + (x >= -2).float() * (-1.90108476182368*torch.sin(1.96884615967697*x) + 1.36805519435098*torch.tanh(0.0422147483765706*x**8 + x - 0.48826253)),
                #     lambda x: 0 * sympy.And(x < -2)+ sympy.And(x >= -2) * (-1.90108476182368*sympy.sin(1.96884615967697*x) + 1.36805519435098*sympy.tanh(0.0422147483765706*x**8 + x - 0.48826253))
                # ),
                # 'custom_159':(
                #     lambda x: (x < -30).float() * torch.zeros_like(x) + ((x >= -30) & (x <= 8)).float() * (-0.00120134839369606*(1 - 0.645860694952566*x)**11 - torch.asinh(x - 1.2626618*torch.sin(1.9301460831063*x)) + 0.20660387) + (x > 8).float() * (-0.43368673*x + 1.9323748/torch.cosh(x)),
                #     lambda x: 0 * sympy.And(x < -30) + (-0.00120134839369606*(1 - 0.645860694952566*x)**11 - sympy.asinh(x - 1.2626618*sympy.sin(1.9301460831063*x)) + 0.20660387) * sympy.And(x >= -30, x <= 8) + (-0.43368673*x + 1.9323748/sympy.cosh(x)) * sympy.And(x > 8)
                # ),
                # 'custom_160':(
                #     lambda x: (x < -30).float() * torch.zeros_like(x) + ((x >= -30) & (x <= 8)).float() * (-torch.tanh(torch.sin(1.86830818715067*x + 0.3942015)) + torch.atan(torch.tan(torch.tanh(x) - torch.asinh(x))/torch.atan(torch.exp(x)))**9) + (x > 8).float() * (0.007758907*x),
                #     lambda x: 0 * sympy.And(x < -30) + (-sympy.tanh(sympy.sin(1.86830818715067*x + 0.3942015)) + sympy.atan(sympy.tan(sympy.tanh(x) - sympy.asinh(x))/sympy.atan(sympy.exp(x)))**9) * sympy.And(x >= -30, x <= 8) + (0.007758907*x) * sympy.And(x > 8)
                # ),
                # 'custom_161':(
                #     lambda x: (x < -10).float() * torch.zeros_like(x) + (x >= -10).float() * (torch.asinh((torch.sin(torch.sin(2.0521882*torch.sin(x) - 2.2573593)) + 0.5365932*torch.asinh(x))/torch.tanh(torch.cos(torch.tanh(x))))),
                #     lambda x: 0 * sympy.And(x < -10) + (sympy.asinh((sympy.sin(sympy.sin(2.0521882*sympy.sin(x) - 2.2573593)) + 0.5365932*sympy.asinh(x))/sympy.tanh(sympy.cos(sympy.tanh(x))))) * sympy.And(x >= -10)
                # ),
                # 'custom_162':(
                #     lambda x: (x < -2).float() * torch.zeros_like(x) + (x >= -2).float() * ((torch.sin(x + 1.2726141) + 0.2505069)*torch.sin(1.3949453*x + 1.31895) - 0.35538423),
                #     lambda x: 0 * sympy.And(x < -2) + sympy.And(x >= -2) * ((sympy.sin(x + 1.2726141) + 0.2505069)*sympy.sin(1.3949453*x + 1.31895) - 0.35538423)
                # ),
                # 'custom_163':(
                #     lambda x: (x < -2).float() * torch.zeros_like(x) + (x >= -2).float() * (x - 0.55007714*torch.sin(2.5323706*x - 5.81796242062632)/torch.sin(torch.tanh(x) + 1.0001391) - 0.8202216),
                #     lambda x: sympy.And(x < -2) * 0 + sympy.And(x >= -2) * (x - 0.55007714*sympy.sin(2.5323706*x - 5.81796242062632)/sympy.sin(sympy.tanh(x) + 1.0001391) - 0.8202216)
                # ),
                # 'custom_164':(
                #     lambda x: (x < -10).float() * torch.zeros_like(x) + ((x >= -10) & (x <= 10)).float() * (-2.3001385*torch.sin(x*torch.atan(torch.cos(torch.cos(x + 0.04829327)) - 0.32902932) - 0.5418271)),
                #     lambda x: 0 * sympy.And(x < -10) + (-2.3001385*sympy.sin(x*sympy.atan(sympy.cos(sympy.cos(x + 0.04829327)) - 0.32902932) - 0.5418271)) * sympy.And(x >= -10, x <= 10)
                # ),
                # 'custom_165':(
                #     lambda x: (x < -300).float() * torch.zeros_like(x) + ((x >= -300) & (x <= 10)).float() * (x + 0.00636071903452418*(0.37058714 - x)**11 - 3.0635624*torch.cos(4.173133*x - torch.sin(x + 0.36493358))) + (x > 10).float() * (-0.0029236139*x),
                #     lambda x: 0 * sympy.And(x < -300) + (x + 0.00636071903452418*(0.37058714 - x)**11 - 3.0635624*sympy.cos(4.173133*x - sympy.sin(x + 0.36493358))) * sympy.And(x >= -300, x <= 10) + (-0.0029236139*x) * sympy.And(x > 10)
                # ),
                # 'custom_166':(
                #     lambda x: (x < -300).float() * torch.zeros_like(x) + ((x >= -300) & (x <= 10)).float() * (3.07866872945681*torch.cos(3.33656557562529*x - 0.15108255) + torch.tan(0.4418895*x - 0.163825914379725)**11 - torch.atan(x)**9) + (x > 10).float() * (0.14792687*x + 4.7998511760094e-8),
                #     lambda x: 0 * sympy.And(x < -300) + (3.07866872945681*sympy.cos(3.33656557562529*x - 0.15108255) + sympy.tan(0.4418895*x - 0.163825914379725)**11 - sympy.atan(x)**9) * sympy.And(x >= -300, x <= 10) + (0.14792687*x + 4.7998511760094e-8) * sympy.And(x > 10)
                # ),
                # 'custom_167':(
                #     lambda x: (x < -50).float() * torch.zeros_like(x) + (x >= -50).float() * (2.22300185367233*torch.atan(x)**10 - 2.22300185367233*torch.sin(3.23895955290957*torch.sin(x) - 0.16736957)/(-0.16998862 + 1/torch.sqrt(x**2 + 1))),
                #     lambda x: 0 * sympy.And(x < -50) + (2.22300185367233*sympy.atan(x)**10 - 2.22300185367233*sympy.sin(3.23895955290957*sympy.sin(x) - 0.16736957)/(-0.16998862 + 1/sympy.sqrt(x**2 + 1))) * sympy.And(x >= -50)
                # ),
                # 'custom_168':(
                #     lambda x: (x < -50).float() * torch.zeros_like(x) + (x >= -50).float() * (((-x + torch.tan(torch.cos(x*(0.15858746 - x))))*(x - 0.95857406) + 1.381145)*torch.cosh(torch.cosh(torch.cos(x)))),
                #     lambda x: 0 * sympy.And(x < -50) + (((-x + sympy.tan(sympy.cos(x*(0.15858746 - x))))*(x - 0.95857406) + 1.381145)*sympy.cosh(sympy.cosh(sympy.cos(x)))) * sympy.And(x >= -50)
                # ),
                # 'custom_169':(
                #     lambda x: (x < -4).float() * torch.zeros_like(x) + (x >= -4).float() * (0.22306156*(x - 0.9199137)*(x + (x - 4.031244)*(-torch.sin(1.1337013*x) - 0.55979633))),
                #     lambda x: 0 * sympy.And(x < -4) + sympy.And(x >= -4) * (0.22306156*(x - 0.9199137)*(x + (x - 4.031244)*(-sympy.sin(1.1337013*x) - 0.55979633)))
                # ),
                # 'custom_170':(
                #     lambda x: (x < -4).float() * torch.zeros_like(x) + (x >= -4).float() * ((0.229049002285451*x - 0.9642367)*((x - 0.7988138)*(torch.sin(1.13675200510265*x) - 0.4629213) - 3.1233044) - 2.1041365),
                #     lambda x: 0 * sympy.And(x < -4) + sympy.And(x >= -4) * ((0.229049002285451*x - 0.9642367)*((x - 0.7988138)*(sympy.sin(1.13675200510265*x) - 0.4629213) - 3.1233044) - 2.1041365)
                # ),
                # 'custom_171':(
                #     lambda x: ((x >= -4) & (x <= 6)).float() * (18.5094180546417*(0.46136677 - torch.tanh(torch.tanh(torch.sin(0.32481205*x + 1.3845815225278))))*torch.sin(1.3114415*x) + 0.25687718) + (x > 6).float() * (-0.17999905*x),
                #     lambda x: sympy.And(x >= -4, x <= 6) * (18.5094180546417*(0.46136677 - sympy.tanh(sympy.tanh(sympy.sin(0.32481205*x + 1.3845815225278))))*sympy.sin(1.3114415*x) + 0.25687718) + sympy.And(x >= 6) * (-0.17999905*x)
                # ),


                # 'custom_141':(
                #     lambda x: torch.where(
                #         x < -100,
                #         torch.zeros_like(x),
                #         -1.081297*torch.sin(torch.asinh(torch.cosh(x))) + 1.081297*torch.sinh(torch.tan(1/torch.sqrt(torch.tan(x + 0.22632751)**2 + 1)) - torch.sinh(torch.atan(x)))
                #     ),
                #     lambda x: sympy.Piecewise(
                #         (sympy.zeros_like(x), x < -100), 
                #         (-1.081297*sympy.sin(sympy.asinh(sympy.cosh(x))) + 1.081297*sympy.sinh(sympy.tan(1/sympy.sqrt(sympy.tan(x + 0.22632751)**2 + 1)) - sympy.sinh(sympy.atan(x))), x >= -100)
                #     )
                # ),
                # 'custom_142':(
                #     lambda x: torch.where(
                #         x < -100,
                #         torch.zeros_like(x),
                #         0.116276457625098*(0.180172760894598*x - 1)**11 - 2.24863342114622*torch.cos(torch.sinh(torch.tanh(x + 0.967478) + torch.atan(x)))
                #     ),
                #     lambda x: sympy.Piecewise(
                #         (sympy.zeros_like(x), x < -100), 
                #         (0.116276457625098*(0.180172760894598*x - 1)**11 - 2.24863342114622*sympy.cos(sympy.sinh(sympy.tanh(x + 0.967478) + sympy.atan(x))), x >= -100)
                #     )
                # ),
                # 'custom_143':(
                #     lambda x: torch.where(
                #         x < -200,
                #         torch.zeros_like(x),
                #         -torch.tan(torch.tanh(x - 0.04783533*torch.asinh(x))**11) + torch.tanh(torch.sin(x - 0.5559367) - 0.1016728)**8
                #     ),
                #     lambda x: sympy.Piecewise(
                #         (sympy.zeros_like(x), x < -200), 
                #         (-sympy.tan(sympy.tanh(x - 0.04783533*sympy.asinh(x))**11) + sympy.tanh(sympy.sin(x - 0.5559367) - 0.1016728)**8, x >= -200)
                #     )
                # ),
                # 'custom_144':(
                #     lambda x: torch.where(
                #         x < -200,
                #         torch.zeros_like(x),
                #         0.297087931525805*x + 0.9105494*torch.tan(torch.asinh(torch.tan(0.9409598*torch.tanh(x)**10))) - 0.014806402818677
                #     ),
                #     lambda x: sympy.Piecewise(
                #         (sympy.zeros_like(x), x < -200), 
                #         (0.297087931525805*x + 0.9105494*sympy.tan(sympy.asinh(sympy.tan(0.9409598*sympy.tanh(x)**10))) - 0.014806402818677, x >= -200)
                #     )
                # ),
                # 'custom_145':(
                #     lambda x: torch.where(
                #         x < -10,
                #         torch.zeros_like(x),
                #         torch.where(
                #             x < 10,
                #             1.1618576*(x - 0.42783698)*torch.sin(1.3623214*torch.sin(x + 0.3873579) - torch.atan(x)) - 0.167689722313248,
                #             -0.6620355*x
                #         )
                #     ),
                #     lambda x: sympy.Piecewise(
                #         (sympy.zeros_like(x), x < -10), 
                #         (1.1618576*(x - 0.42783698)*sympy.sin(1.3623214*sympy.sin(x + 0.3873579) - sympy.atan(x)) - 0.167689722313248, (x >= -10) & (x <= 10)), 
                #         (-0.6620355*x, x > 10)
                #     )
                # ),
                # 'custom_146':(
                #     lambda x: torch.where(
                #         x < -10,
                #         torch.zeros_like(x),
                #         torch.where(
                #             x < 10,
                #             51.9513878899523*(0.644730319300757*torch.sin(0.3801634*x + 1.00676369431796) - 1)**9 + torch.sin(2.50877148044637*x - 0.739856679404619) + 0.54830927,
                #             0.288082612238201*x
                #         )
                #     ),
                #     lambda x: sympy.Piecewise(
                #         (sympy.zeros_like(x), x < -200), 
                #         (51.9513878899523*(0.644730319300757*sympy.sin(0.3801634*x + 1.00676369431796) - 1)**9 + sympy.sin(2.50877148044637*x - 0.739856679404619) + 0.54830927, (x >= -10) & (x <= 10)), 
                #         (0.288082612238201*x, x > 10)
                #     )
                # ),
                # 'custom_147':(
                #     lambda x: torch.where(
                #         x < -200,
                #         torch.zeros_like(x),
                #         torch.where(
                #             x < 10,
                #             -torch.sin(x*(torch.sin(x) + 0.8326211)) + torch.exp(-7.405453*torch.sin(torch.asinh(x) + 1.1280764)),
                #             -0.092607185*x
                #         )
                #     ),
                #     lambda x: sympy.Piecewise(
                #         (sympy.zeros_like(x), x < -200), 
                #         (-sympy.sin(x*(sympy.sin(x) + 0.8326211)) + sympy.exp(-7.405453*sympy.sin(sympy.asinh(x) + 1.1280764)), (x >= -10) & (x <= 10)), 
                #         (-0.092607185*x, x > 10)
                #     )
                # ),
                # 'custom_148':(
                #     lambda x: torch.where(
                #         x < -10,
                #         torch.zeros_like(x),
                #         (x + 1.2887747)*(torch.sin(torch.sin(x)) - 0.673435)**9 - torch.atan(38.0709491365401*x**11) + 1.5454899
                #     ),
                #     lambda x: sympy.Piecewise(
                #         (sympy.zeros_like(x), x < -10), 
                #         ((x + 1.2887747)*(sympy.sin(sympy.sin(x)) - 0.673435)**9 - sympy.atan(38.0709491365401*x**11) + 1.5454899, x >= -10)
                #     )
                # ),
                # 'custom_149':(
                #     lambda x: torch.where(
                #         x < -10,
                #         torch.zeros_like(x),
                #         (torch.sin(0.76567507*x) - 0.3282788)**11 + 1.49476863000315e-6*torch.exp(4.4306417*x) - torch.cos(x)**8
                #     ),
                #     lambda x: sympy.Piecewise(
                #         (sympy.zeros_like(x), x < -10), 
                #         ((sympy.sin(0.76567507*x) - 0.3282788)**11 + 1.49476863000315e-6*sympy.exp(4.4306417*x) - sympy.cos(x)**8, x >= -10)
                #     )
                # ),
                # 'custom_151':(
                #     lambda x: torch.where(
                #         x < -50,
                #         torch.zeros_like(x),
                #         -8.19227626152316e-5*(0.724470527959346*x - 1)**10 + torch.sin(2.06217158628631*x) + torch.sin(x - 0.2726203) + 0.32098338
                #     ),
                #     lambda x: sympy.Piecewise(
                #         (sympy.zeros_like(x), x < -50), 
                #         (-8.19227626152316e-5*(0.724470527959346*x - 1)**10 + sympy.sin(2.06217158628631*x) + sympy.sin(x - 0.2726203) + 0.32098338, x >= -50)
                #     )
                # ),
                # 'custom_152':(
                #     lambda x: torch.where(
                #         x < -50,
                #         torch.zeros_like(x),
                #         0.00145503867552248*(0.706639571278946*x - 1)**8 - torch.sin(2.0351255*x) - torch.tanh(torch.sin(x)*torch.atan(x + 1.7585657))
                #     ),
                #     lambda x: sympy.Piecewise(
                #         (sympy.zeros_like(x), x < -50), 
                #         (0.00145503867552248*(0.706639571278946*x - 1)**8 - sympy.sin(2.0351255*x) - sympy.tanh(sympy.sin(x)*sympy.atan(x + 1.7585657)), x >= -50)
                #     )
                # ),
                # 'custom_153':(
                #     lambda x: torch.where(
                #         x < -20,
                #         torch.zeros_like(x),
                #         torch.where(
                #             x < 8,
                #             torch.tan(torch.tan(torch.sin(torch.sin(torch.tan(0.27664214*x + 0.0154092659715112))*torch.atan(torch.tanh(x - 1.206411))))),
                #             0.3004957*x - 2.573844*torch.exp(-x)
                #         )
                #     ),
                #     lambda x: sympy.Piecewise(
                #         (sympy.zeros_like(x), x < -120), 
                #         (torch.tan(torch.tan(torch.sin(torch.sin(torch.tan(0.27664214*x + 0.0154092659715112))*torch.atan(torch.tanh(x - 1.206411))))), (x >= -20) & (x <= 8)), 
                #         (0.3004957*x - 2.573844*torch.exp(-x), x > 8)
                #     )
                # ),
                # 'custom_154':(
                #     lambda x: torch.where(
                #         x < -120,
                #         torch.zeros_like(x),
                #         torch.where(
                #             x < 8,
                #             0.604127033416079*torch.cos(torch.cosh(x/torch.cosh(torch.atan(torch.exp(torch.tan(torch.sin(0.54478717*x) - 0.23112519))))))*torch.tanh(x),
                #             -0.15891315*x + 1.38558071656444*torch.exp(-x)
                #         )
                #     ),
                #     lambda x: sympy.Piecewise(
                #         (sympy.zeros_like(x), x < -120), 
                #         (0.604127033416079*sympy.cos(sympy.cosh(x/sympy.cosh(sympy.atan(sympy.exp(sympy.tan(sympy.sin(0.54478717*x) - 0.23112519))))))*sympy.tanh(x), (x >= -20) & (x <= 8)), 
                #         (-0.15891315*x + 1.38558071656444*sympy.exp(-x), x > 8)
                #     )
                # ),
                # 'custom_155':(
                #     lambda x: torch.where(
                #         x < -10,
                #         torch.zeros_like(x),
                #         -torch.cos(x - 0.87358713) + 0.34094578 + 2.03447110180282*torch.exp(-x)*torch.tanh(x)**11
                #     ),
                #     lambda x: sympy.Piecewise(
                #         (sympy.zeros_like(x), x < -10), 
                #         (-sympy.cos(x - 0.87358713) + 0.34094578 + 2.03447110180282*sympy.exp(-x)*sympy.tanh(x)**11, x >= -10)
                #     )
                # ),
                # 'custom_156':(
                #     lambda x: torch.where(
                #         x < -10,
                #         torch.zeros_like(x),
                #         torch.sin(torch.asinh(1.4513612*x + 0.286566639097772)) + torch.exp(-3.360346*torch.atan(x + 1.2375063))
                #     ),
                #     lambda x: sympy.Piecewise(
                #         (sympy.zeros_like(x), x < -10), 
                #         (sympy.sin(sympy.asinh(1.4513612*x + 0.286566639097772)) + sympy.exp(-3.360346*sympy.atan(x + 1.2375063)), x >= -10)
                #     )
                # ),
                # 'custom_157':(
                #     lambda x: torch.where(
                #         x < -10,
                #         torch.zeros_like(x),
                #         4.7890716*torch.tanh(torch.sin(1.3862643*x - 0.6897017) + 0.938898)*torch.atan(torch.cos(x + 0.2100102)) - 0.51874393
                #     ),
                #     lambda x: sympy.Piecewise(
                #         (sympy.zeros_like(x), x < -10), 
                #         (4.7890716*sympy.tanh(sympy.sin(1.3862643*x - 0.6897017) + 0.938898)*sympy.atan(sympy.cos(x + 0.2100102)) - 0.51874393, x >= -10)
                #     )
                # ),
                # 'custom_158':(
                #     lambda x: torch.where(
                #         x < -10,
                #         torch.zeros_like(x),
                #         -2.07186053511144*torch.cos(1.5264728*x - 0.7201182) + torch.cos(torch.tan(torch.cos(1.1874254*x - 0.8902031))) + 0.20628148
                #     ),
                #     lambda x: sympy.Piecewise(
                #         (sympy.zeros_like(x), x < -10), 
                #         (-2.07186053511144*sympy.cos(1.5264728*x - 0.7201182) + sympy.cos(sympy.tan(sympy.cos(1.1874254*x - 0.8902031))) + 0.20628148, x >= -10)
                #     )
                # ),
                # 'custom_159':(
                #     lambda x: torch.where(
                #         x < -30,
                #         torch.zeros_like(x),
                #         torch.where(
                #             x < 8,
                #             -0.00120134839369606*(1 - 0.645860694952566*x)**11 - torch.asinh(x - 1.2626618*torch.sin(1.9301460831063*x)) + 0.20660387,
                #             -0.43368673*x + 1.9323748/torch.cosh(x)
                #         )
                #     ),
                #     lambda x: sympy.Piecewise(
                #         (sympy.zeros_like(x), x < -30), 
                #         (-0.00120134839369606*(1 - 0.645860694952566*x)**11 - sympy.asinh(x - 1.2626618*sympy.sin(1.9301460831063*x)) + 0.20660387, (x >= -30) & (x <= 8)), 
                #         (-0.43368673*x + 1.9323748/sympy.cosh(x), x > 8)
                #     )
                # ),
                # 'custom_160':(
                #     lambda x: torch.where(
                #         x < -30,
                #         torch.zeros_like(x),
                #         torch.where(
                #             x < 8,
                #             -torch.tanh(torch.sin(1.86830818715067*x + 0.3942015)) + torch.atan(torch.tan(torch.tanh(x) - torch.asinh(x))/torch.atan(torch.exp(x)))**9,
                #             0.007758907*x
                #         )
                #     ),
                #     lambda x: sympy.Piecewise(
                #         (sympy.zeros_like(x), x < -30), 
                #         (-sympy.tanh(sympy.sin(1.86830818715067*x + 0.3942015)) + sympy.atan(sympy.tan(sympy.tanh(x) - sympy.asinh(x))/sympy.atan(sympy.exp(x)))**9, (x >= -30) & (x <= 8)), 
                #         (0.007758907*x, x > 8)
                #     )
                # ),
                # 'custom_161':(
                #     lambda x: torch.where(
                #         x < -10,
                #         torch.zeros_like(x),
                #         torch.asinh((torch.sin(torch.sin(2.0521882*torch.sin(x) - 2.2573593)) + 0.5365932*torch.asinh(x))/torch.tanh(torch.cos(torch.tanh(x))))
                #     ),
                #     lambda x: sympy.Piecewise(
                #         (sympy.zeros_like(x), x < -10), 
                #         (sympy.asinh((sympy.sin(sympy.sin(2.0521882*sympy.sin(x) - 2.2573593)) + 0.5365932*sympy.asinh(x))/sympy.tanh(sympy.cos(sympy.tanh(x)))), x >= -10)
                #     )
                # ),
                # 'custom_162':(
                #     lambda x: torch.where(
                #         x < -10,
                #         torch.zeros_like(x),
                #         (torch.tan(torch.cos(1.96596119066788*x - 0.3942599)) + 0.06856037)/(torch.sqrt((torch.tan(torch.cos(1.96596119066788*x - 0.3942599)) + 0.06856037)**2/torch.cosh(x - 0.37355217)**2 + 1)*torch.cosh(x - 0.37355217))
                #     ),
                #     lambda x: sympy.Piecewise(
                #         (sympy.zeros_like(x), x < -10), 
                #         ((sympy.tan(sympy.cos(1.96596119066788*x - 0.3942599)) + 0.06856037)/(sympy.sqrt((sympy.tan(sympy.cos(1.96596119066788*x - 0.3942599)) + 0.06856037)**2/sympy.cosh(x - 0.37355217)**2 + 1)*sympy.cosh(x - 0.37355217)), x >= -10)
                #     )
                # ),
                # 'custom_163':(
                #     lambda x: torch.where(
                #         x < -10,
                #         torch.zeros_like(x),
                #         torch.where(
                #             x < 0,
                #             (torch.asinh(x) + 0.05774226)**9 - torch.asinh(1.09633457783544*torch.cos(3.008334*x) + 0.962837676720023) + 0.3177639,
                #             x - torch.cosh(2.2112162*torch.cos(x) + torch.atan(0.683457765932391*x*torch.cosh(x)) - 0.9476219) + 0.7621713
                #         )
                #     ),
                #     lambda x: sympy.Piecewise(
                #         (sympy.zeros_like(x), x < -10), 
                #         ((sympy.asinh(x) + 0.05774226)**9 - sympy.asinh(1.09633457783544*sympy.cos(3.008334*x) + 0.962837676720023) + 0.3177639, (x >= -10) & (x <= 0)), 
                #         (x - sympy.cosh(2.2112162*sympy.cos(x) + sympy.atan(0.683457765932391*x*sympy.cosh(x)) - 0.9476219) + 0.7621713, x > 0)
                #     )
                # ),
                # 'custom_164':(
                #     lambda x: torch.where(
                #         x < -10,
                #         torch.zeros_like(x),
                #         -2.3001385*torch.sin(x*torch.atan(torch.cos(torch.cos(x + 0.04829327)) - 0.32902932) - 0.5418271)
                #     ),
                #     lambda x: sympy.Piecewise(
                #         (sympy.zeros_like(x), x < -10), 
                #         (-2.3001385*sympy.sin(x*sympy.atan(sympy.cos(sympy.cos(x + 0.04829327)) - 0.32902932) - 0.5418271), (x >= -10) & (x <= 10)), 
                #     )
                # ),
                # 'custom_165':(
                #     lambda x: torch.where(
                #         x < -300,
                #         torch.zeros_like(x),
                #         torch.where(
                #             x < 10,
                #             x + 0.00636071903452418*(0.37058714 - x)**11 - 3.0635624*torch.cos(4.173133*x - torch.sin(x + 0.36493358)),
                #             -0.0029236139*x
                #         )
                #     ),
                #     lambda x: sympy.Piecewise(
                #         (sympy.zeros_like(x), x < -300), 
                #         (x + 0.00636071903452418*(0.37058714 - x)**11 - 3.0635624*torch.cos(4.173133*x - torch.sin(x + 0.36493358)), (x >= -300) & (x <= 10)), 
                #         (-0.0029236139*x, x > 10)
                #     )
                # ),
                # 'custom_166':(
                #     lambda x: torch.where(
                #         x < -300,
                #         torch.zeros_like(x),
                #         torch.where(
                #             x < 10,
                #             3.07866872945681*torch.cos(3.33656557562529*x - 0.15108255) + torch.tan(0.4418895*x - 0.163825914379725)**11 - torch.atan(x)**9,
                #             0.14792687*x + 4.7998511760094e-8
                #         )
                #     ),
                #     lambda x: sympy.Piecewise(
                #         (sympy.zeros_like(x), x < -300), 
                #         (3.07866872945681*sympy.cos(3.33656557562529*x - 0.15108255) + sympy.tan(0.4418895*x - 0.163825914379725)**11 - sympy.atan(x)**9, (x >= -300) & (x <= 10)), 
                #         (0.14792687*x + 4.7998511760094e-8, x > 10)
                #     )
                # ),
                # 'custom_167':(
                #     lambda x: torch.where(
                #         x < -50,
                #         torch.zeros_like(x),
                #         2.22300185367233*torch.atan(x)**10 - 2.22300185367233*torch.sin(3.23895955290957*torch.sin(x) - 0.16736957)/(-0.16998862 + 1/torch.sqrt(x**2 + 1))
                #     ),
                #     lambda x: sympy.Piecewise(
                #         (sympy.zeros_like(x), x < -50), 
                #         (2.22300185367233*sympy.atan(x)**10 - 2.22300185367233*sympy.sin(3.23895955290957*sympy.sin(x) - 0.16736957)/(-0.16998862 + 1/sympy.sqrt(x**2 + 1)), x >= -50)
                #     )
                # ),
                # 'custom_168':(
                #     lambda x: torch.where(
                #         x < -50,
                #         torch.zeros_like(x),
                #         ((-x + torch.tan(torch.cos(x*(0.15858746 - x))))*(x - 0.95857406) + 1.381145)*torch.cosh(torch.cosh(torch.cos(x)))
                #     ),
                #     lambda x: sympy.Piecewise(
                #         (sympy.zeros_like(x), x < -50), 
                #         (((-x + sympy.tan(sympy.cos(x*(0.15858746 - x))))*(x - 0.95857406) + 1.381145)*sympy.cosh(sympy.cosh(sympy.cos(x))), x >= -50)
                #     )
                # ),
                # 'custom_169':(
                #     lambda x: torch.where(
                #         x < -10,
                #         torch.zeros_like(x),
                #         torch.where(
                #             x < 0,
                #             2.5336103*(torch.asinh(torch.cosh(2*x)) - 0.5175451)*torch.sin(torch.tan(torch.asinh(torch.sin(x)) + 0.45229566)) + 0.076056175,
                #             torch.where(
                #                 x < 7.3,
                #                 -0.583633831401976*x + torch.sin(4.47601605403758e-9*x**10) + 0.65338355,
                #                 -x + (x - torch.tan(torch.cos(x)) - 0.17235969)*torch.sin(1.11121951674841*x) - 1.8452086
                #             )
                #         )
                #     ),
                #     lambda x: sympy.Piecewise(
                #         (sympy.zeros_like(x), x < -10), 
                #         (2.5336103*(sympy.asinh(sympy.cosh(2*x)) - 0.5175451)*sympy.sin(sympy.tan(sympy.asinh(sympy.sin(x)) + 0.45229566)) + 0.076056175, (x >= -10) & (x <= 0)), 
                #         (-0.583633831401976*x + sympy.sin(4.47601605403758e-9*x**10) + 0.65338355, (x > 0) & (x <= 7.3)),
                #         (-x + (x - sympy.tan(sympy.cos(x)) - 0.17235969)*sympy.sin(1.11121951674841*x) - 1.8452086, x > 7.3)
                #     )
                # ),
                # 'custom_170':(
                #     lambda x: torch.where(
                #         x < -10,
                #         torch.zeros_like(x),
                #         torch.where(
                #             x < 0,
                #             (5.924705*torch.sqrt(x**2 + 1) - 4.8792624)*torch.atan(torch.sin(x) + 0.49543446) + 0.02248232,
                #             torch.where(
                #                 x < 7.3,
                #                 torch.cos(torch.tan(0.121708396418852*x)) + torch.tan(torch.cos(torch.tan(torch.cos(0.45920664*x)) - 8.359178)) - 1.6888623,
                #                 -x + (x - torch.tan(torch.cos(x)) - 0.14064497)*torch.sin(1.11123713774961*x) - 1.88951
                #             )
                #         )
                #     ),
                #     lambda x: sympy.Piecewise(
                #         (sympy.zeros_like(x), x < -10), 
                #         ((5.924705*sympy.sqrt(x**2 + 1) - 4.8792624)*sympy.atan(sympy.sin(x) + 0.49543446) + 0.02248232, (x >= -10) & (x <= 0)), 
                #         (sympy.cos(sympy.tan(0.121708396418852*x)) + sympy.tan(sympy.cos(sympy.tan(sympy.cos(0.45920664*x)) - 8.359178)) - 1.6888623, (x > 0) & (x <= 7.3)),
                #         (-x + (x - sympy.tan(sympy.cos(x)) - 0.14064497)*sympy.sin(1.11123713774961*x) - 1.88951, x > 7.3)
                #     )
                # ),
                # 'custom_171':(
                #     lambda x: torch.where(
                #         x < 1,
                #         -x + torch.sin(1.823935*x - 0.16595160875954)*torch.asinh(torch.tan(torch.asinh(0.5993536*x)) - 3.6428304),
                #         torch.where(
                #             x < 10,
                #             1.0969447*x + 0.90119827*torch.cos(2.19582296502326*x)*torch.cosh(0.965306964236053*x - 1.9234264458778) - 3.153261,
                #             torch.where(
                #                 x < 40,
                #                 -0.179999050001013*x,
                #                 -0.17999905*x
                #             )
                #         )
                #     ),
                #     lambda x: sympy.Piecewise(
                #         (-x + sympy.sin(1.823935*x - 0.16595160875954)*sympy.asinh(sympy.tan(sympy.asinh(0.5993536*x)) - 3.6428304), (x >= -10) & (x <= 1)), 
                #         (1.0969447*x + 0.90119827*sympy.cos(2.19582296502326*x)*sympy.cosh(0.965306964236053*x - 1.9234264458778) - 3.153261, (x > 1) & (x <= 10)), 
                #         (-0.179999050001013*x, (x >= 20) & (x <= 40)), 
                #         (-0.17999905*x, x > 40)
                #     )
                # ),




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
    
