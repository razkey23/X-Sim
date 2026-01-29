#ifndef MEMRISTOR_H_
#define MEMRISTOR_H_

class Memristor {
public:
    virtual double ApplyVoltage(double V_applied, double dt) = 0;
    virtual double GetResistance(double V_applied) = 0;
    virtual void SetWeight(int weight) = 0;
};

#endif  // MEMRISTOR_H_