//
// Copyright (C) 2016 David Eckhoff <david.eckhoff@fau.de>
//
// Documentation for these modules is at http://veins.car2x.org/
//
// SPDX-License-Identifier: GPL-2.0-or-later
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//

#include "gradient_simple/GradientSimpleApp.h"

#include <string>

#include "gradient_simple/GradientSimpleMessage_m.h"

using namespace veins;

Define_Module(veins::GradientSimpleApp);

void GradientSimpleApp::initialize(int stage)
{
    DemoBaseApplLayer::initialize(stage);
    if (stage == 0) {
        // Initializing members and pointers of your application goes here
        EV << "Initializing " << par("appName").stringValue() << std::endl;
    }
    else if (stage == 1) {
        // Initializing members that require initialized other modules goes here
        std::string node_name = findHost()->getFullName();
        if (node_name == "node[0]") {
          gradientHash = myId + 1;
          findHost()->getDisplayString().setTagArg("i", 1, "green");
        } else {
          gradientHash = 0;
        }
        EV << "Initialized " << node_name << " with vehicle ID " << myId
           << " and gradientHash " << gradientHash << std::endl;
    }
}

void GradientSimpleApp::finish()
{
    DemoBaseApplLayer::finish();
    // statistics recording goes here
}

void GradientSimpleApp::onBSM(DemoSafetyMessage* bsm)
{
    // Your application has received a beacon message from another car or RSU
    // code for handling the message goes here
    EV << findHost()->getFullName() << " received BSM with position " << bsm->getSenderPos() << std::endl;
    
    if (gradientHash > 0) {
        GradientSimpleMessage* wsm = new GradientSimpleMessage("gradients");  // Make message green
        populateWSM(wsm);
        wsm->setSenderAddress(myId);
        wsm->setGradientHash(gradientHash);
        sendDown(wsm);
    }
}

void GradientSimpleApp::onWSM(BaseFrame1609_4* wsm)
{
    // Your application has received a data message from another car or RSU
    // code for handling the message goes here, see TraciDemo11p.cc for examples
    if (GradientSimpleMessage* tvm = dynamic_cast<GradientSimpleMessage*>(wsm)) {
        // Set vehicle to green
        findHost()->getDisplayString().setTagArg("i", 1, "green");
        uint32_t senderGradientHash = tvm->getGradientHash();
        EV << findHost()->getFullName() << " received WSM from senderAddress " << tvm->getSenderAddress()
           << " with gradientHash " << senderGradientHash << std::endl;
        gradientHash = senderGradientHash;
    }
}

void GradientSimpleApp::onWSA(DemoServiceAdvertisment* wsa)
{
    // Your application has received a service advertisement from another car or RSU
    // code for handling the message goes here, see TraciDemo11p.cc for examples
}

void GradientSimpleApp::handleSelfMsg(cMessage* msg)
{
    DemoBaseApplLayer::handleSelfMsg(msg);
    // this method is for self messages (mostly timers)
    // it is important to call the DemoBaseApplLayer function for BSM and WSM transmission
}

void GradientSimpleApp::handlePositionUpdate(cObject* obj)
{
    DemoBaseApplLayer::handlePositionUpdate(obj);
    // the vehicle has moved. Code that reacts to new positions goes here.
    // member variables such as currentPosition and currentSpeed are updated in the parent class
}
